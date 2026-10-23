"""Causal tracking, stale-state rejection, partition guard and scoring independence."""

import json
import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from scripts.assess_navigation_signal_crops import crop_feature
from src.navigation.contracts import VIEWS, CameraCalibration, SensorPacket
from src.navigation.policy_worker import PrivilegeGuard
from src.navigation.signal_perception import (
    FrozenSignalPerception,
    state_features,
    state_probabilities,
)
from src.navigation.signal_temporal_audit import episode_scope, score_frame, summarize_frames
from src.navigation.signal_tracking import SignalTracker, angular_box


def packet(index, heading=0):
    c = CameraCalibration(512, 288, 100, (1.3, 0, 2.3), (0, 0, 0))
    return SensorPacket(
        index * 2,
        index / 10,
        {v: np.zeros((288, 512, 3), np.uint8) for v in VIEWS},
        dict.fromkeys(VIEWS, c),
        np.array([[2, 0]], np.float32),
        np.array([0, heading, 0], np.float32),
    )


def observation(probabilities=(1, 0, 0), x=256):
    return dict(
        view="front", box=[x - 2, 20, x + 2, 30], score=0.9, probabilities=list(probabilities)
    )


def test_rgb_crop_export_keeps_existing_rounding_features_and_class_decisions():
    image = np.random.default_rng(1).integers(256, size=(288, 512, 3), dtype=np.uint8)
    assert np.array_equal(
        state_features(image, [10.8, 20.2, 14.1, 31.9]), crop_feature(image, [10, 20, 15, 32])
    )
    linear = dict(coef=np.eye(3), intercept=np.array([1000, 999, 998]), classes=np.array([0, 1, 2]))
    p = state_probabilities(linear, np.array([0, 3, 0]))
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1) and p.argmax() == 1
    with pytest.raises(ValueError):
        state_features(image, [600, 20, 601, 30])


def test_temporal_confirmation_stale_unknown_expiry_and_new_track_no_inherited_state():
    tracker = SignalTracker()
    a = tracker.step(packet(0), [observation()])["observations"][0]
    assert a["tracked_state"] == "Unknown" and a["applicability"] == "unresolved"
    b = tracker.step(packet(1), [observation()])["observations"][0]
    assert b["track_id"] == a["track_id"] and b["tracked_state"] == "Red"
    for n in (2, 3):
        assert tracker.step(packet(n), [])["tracks"][0]["state"] == "Red"
    assert tracker.step(packet(4), [])["tracks"][0]["state"] == "Unknown"
    for n in (5, 6):
        tracker.step(packet(n), [])
    assert tracker.step(packet(7), [])["tracks"] == []
    green = tracker.step(packet(8), [observation((0, 0, 1))])["observations"][0]
    assert green["track_id"] != a["track_id"] and green["tracked_state"] == "Unknown"


def test_association_ignores_color_and_tracks_geometry_when_detections_reorder():
    tracker = SignalTracker()
    first = tracker.step(packet(0), [observation(x=200), observation((0, 0, 1), x=300)])
    second = tracker.step(packet(1), [observation(x=300), observation((0, 0, 1), x=200)])
    assert [r["track_id"] for r in second["observations"]] == [2, 1]
    assert first["observations"][0]["track_id"] == 1
    # EMA delays a hard red->green switch; no future state changes earlier outputs.
    before = second["observations"][1]["tracked_state"]
    for n in range(2, 6):
        third = tracker.step(packet(n), [observation((0, 0, 1), x=200)])
    assert before == "Red" and third["observations"][0]["tracked_state"] == "Green"


def test_heading_wrap_compensation_time_guard_and_invalid_input_do_not_advance_memory():
    tracker = SignalTracker()
    first = tracker.step(packet(0, math.pi - 0.01), [observation()])
    second = tracker.step(packet(1, -math.pi + 0.01), [observation()])
    assert first["observations"][0]["track_id"] == second["observations"][0]["track_id"]
    with pytest.raises(ValueError, match="contiguous"):
        tracker.step(packet(3), [])
    with pytest.raises(ValueError, match="Invalid RGB"):
        tracker.step(packet(2), [observation((-1, 1, 1))])
    tracker.step(packet(2), [])
    with pytest.raises(ValueError, match="Snapshot"):
        tracker.snapshot(0.1)
    c = replace(packet(0).calibration["front"], pitch_yaw_roll=(1, 0, 0))
    with pytest.raises(ValueError, match="pitch/roll"):
        angular_box([10, 10, 20, 20], c, 0)


def test_internal_train_and_component_validation_overlap_are_independently_guarded():
    guard = dict(allowed_junctions={"Town03_Opt": [238, 655]})
    roles = {("Town03_Opt", 655): "validation"}
    meta = dict(town="Town03_Opt", partition="train", junction_ids=[238])
    assert episode_scope(meta, guard, roles)["eligible"]
    assert not episode_scope(dict(meta, junction_ids=[238, 655]), guard, roles)["eligible"]
    for change in (dict(town="Town05"), dict(partition="heldout_dev"), dict(junction_ids=[777])):
        with pytest.raises(ValueError, match="TRAIN"):
            episode_scope(dict(meta, **change), guard, roles)
    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.signal_temporal_audit")


def test_scores_share_matches_unknowns_not_silent_drops_and_no_applicability_claim():
    row = dict(
        next_light_state="Red",
        signal=dict(
            recorded_actor_id=7,
            views={"front": dict(boxes=[dict(in_image=True, bbox_xyxy=[254, 20, 258, 30])])},
        ),
    )
    tracker = SignalTracker()
    predicted = tracker.step(packet(0), [observation()])["observations"]
    scored = score_frame(row, predicted)
    result = summarize_frames([dict(frame=0, observations=predicted, scored=scored)])
    assert result["counts"]["matched_housings"] == 1
    assert result["counts"]["raw_state_correct"] == 1
    assert result["counts"]["tracked_argmax_correct"] == 1
    assert result["counts"]["tracked_unknown"] == 1
    assert predicted[0]["applicability"] == "unresolved"


def test_bundle_rejects_changed_artifact_before_deserializing_weights(tmp_path):
    artifact = tmp_path / "weights.pt"
    artifact.write_bytes(b"changed checkpoint")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            dict(
                detector_architecture="signal-detector-roi-residual-v1",
                expanded_state_accepted=True,
                detector_checkpoint=str(artifact),
                detector_sha256="0" * 64,
            )
        )
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenSignalPerception(bundle, "cpu")


def test_inference_modules_import_under_fresh_process_privilege_guard():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from src.navigation.policy_worker import PrivilegeGuard, FORBIDDEN
sys.meta_path.insert(0, PrivilegeGuard())
from src.navigation.signal_tracking import SignalTracker
from src.navigation.signal_perception import FrozenSignalPerception
assert not any(name in sys.modules for name in FORBIDDEN)
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

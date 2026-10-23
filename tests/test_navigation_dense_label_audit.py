"""The label auditor rejects copied-RGB corruption instead of trusting collector claims."""

import json
from dataclasses import replace

import numpy as np
import pytest

import scripts.audit_navigation_dense_labels as audit
from scripts.navigation_label_rig import LabelShards
from src.navigation.dagger import digest
from tests.test_navigation_collection import packet


def test_independent_label_rgb_match_catches_wrong_original_policy_pixels(tmp_path, monkeypatch):
    output = tmp_path / "native"
    episode = output / "episodes" / "one"
    (episode / "data").mkdir(parents=True)
    (output / "recordings").mkdir()
    (output / "recordings" / "native.log").write_bytes(b"fixture")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(catalog="fixture")))
    item = dict(route_id=1, town="Town01_Opt", role="train")
    (episode / "data" / "episode.json").write_text(
        json.dumps(
            dict(dense_label_role="train", label_plan_sha256=digest(plan), town="Town01_Opt")
        )
    )
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                student=None,
                final_benchmark=False,
                rule_test=None,
                training_collection=True,
                collection=dict(dense_labels=True),
            )
        )
    )
    (episode / "world_trace.jsonl").write_text(
        json.dumps(dict(frame=10, actors=[dict(id=1, role="hero")])) + "\n"
    )
    (episode / "label_alignment.jsonl").write_text("{}\n")
    rgb = {view: np.zeros((288, 512, 3), np.uint8) for view in ("front", "left", "right")}
    labels = {
        f"{v}_{k}": np.zeros((288, 512, 4), np.uint8) for v in rgb for k in ("instance", "depth")
    }
    writer = LabelShards(episode / "labels", True)
    for f in (10, 20):
        writer.append(f, f / 20, rgb, labels)
    writer.close()
    (episode / "label_profile.json").write_text(
        json.dumps(
            dict(
                label_role="train",
                label_timestamps=2,
                maximum_position_error_m=0.0,
                maximum_angle_error_degrees=0.0,
                all_raw_pixel_roundtrips_verified=True,
            )
        )
    )
    shard = episode / "data" / "shard_000000.zip"
    import zipfile

    with zipfile.ZipFile(shard, "x") as z:
        for frame in (10, 20):
            z.writestr(
                f"{frame}/targets.json",
                json.dumps(
                    dict(
                        facts=dict(actuator_feedback_frame=frame + 1),
                        executed_control=[0.0, 0.0, 1.0],
                    )
                ),
            )
    monkeypatch.setattr(audit, "validate_label_guard", lambda *a: None)
    monkeypatch.setattr(audit, "actual_geography", lambda *a: dict(fixture_only=True))
    monkeypatch.setattr(audit, "vehicle_frames", lambda *a: [])
    monkeypatch.setattr(
        audit,
        "align_world_frames",
        lambda *a: ({f: dict(control=[0.0, 0.0, 1.0]) for f in (11, 21)}, dict()),
    )
    monkeypatch.setattr(
        audit, "recorder_contacts", lambda *a: dict(complete=True, ego_contact_starts=0)
    )
    monkeypatch.setattr(
        audit, "shard_records", lambda *a: [dict(frame=f, prefix=str(f)) for f in (10, 20)]
    )

    def load(path, row):
        base = packet(row["frame"])
        return replace(base, rgb=rgb)

    monkeypatch.setattr(audit, "load_policy_sample", load)
    assert audit.audit_episode(item, output, plan, lambda: None)["all_label_RGB_matches_original"]
    wrong = {v: a.copy() for v, a in rgb.items()}
    wrong["left"][0, 0, 0] = 1
    monkeypatch.setattr(
        audit, "load_policy_sample", lambda p, row: replace(load(p, row), rgb=wrong)
    )
    with pytest.raises(ValueError, match="differs from independent"):
        audit.audit_episode(item, output, plan, lambda: None)

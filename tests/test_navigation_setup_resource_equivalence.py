import copy
import hashlib
import json

import pytest
import yaml

from scripts import certify_navigation_rule_pairs as certifier


def test_only_approved_disk_floor_can_reuse_physical_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(certifier, "ROOT", tmp_path)
    configuration = dict(runtime=dict(minimum_free_gb=25, physics_hz=20),
                         camera=dict(width=512, height=288, fov=100))
    config = tmp_path / "configs/navigation.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump(configuration))
    reference = dict(config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                     configuration=copy.deepcopy(configuration))
    assert certifier.resource_only_config_change(reference) is None
    configuration["runtime"]["minimum_free_gb"] = 10
    config.write_text(yaml.safe_dump(configuration))
    plan = tmp_path / "reports/navigation_local_storage_plan_20261020.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps(dict(user_approved=True, minimum_free_bytes_proposed=10**10)))
    guard = tmp_path / "outputs/navigation_compact_execution_20261020/storage_guard/request.json"
    guard.parent.mkdir(parents=True)
    guard.write_text(json.dumps(dict(plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest())))
    proof = certifier.resource_only_config_change(reference)
    assert proof["only_changed_key"] == "runtime.minimum_free_gb"
    assert proof["all_sensor_and_physics_values_identical"]
    for change in (("camera", "fov", 110), ("runtime", "physics_hz", 10)):
        bad = copy.deepcopy(configuration)
        bad[change[0]][change[1]] = change[2]
        config.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError, match="beyond approved storage-only"):
            certifier.resource_only_config_change(reference)

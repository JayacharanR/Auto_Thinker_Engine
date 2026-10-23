import json
from pathlib import Path

import pytest

from src.navigation.dagger import digest
from src.navigation.rule_case_contract import verify_rule_case


@pytest.fixture
def prepared(tmp_path):
    original = Path("outputs/navigation_rule_cases_v3_20261019/signal0_empty_light_a0")
    if not original.exists():
        pytest.skip("Optional prepared native rule geometry unavailable")
    (tmp_path / "route.xml").write_bytes((original / "route.xml").read_bytes())
    path = tmp_path / "case.json"
    path.write_bytes((original / "case.json").read_bytes())
    path.with_suffix(".sha256").write_text(digest(path))
    return path


def check(path, **changed):
    options = dict(
        path=path,
        route=path.parent / "route.xml",
        subset="",
        seed=61001,
        collection=None,
        final=None,
        overrides=False,
        max_steps=0,
    )
    return verify_rule_case(**dict(options, **changed))


def test_physical_case_rejects_collection_final_override_and_seed_drift(prepared):
    assert check(prepared)["setup_only_component_test"]
    for changed in (
        dict(seed=42),
        dict(collection={"mode": "expert"}),
        dict(final="registry.json"),
        dict(overrides=True),
        dict(max_steps=100),
        dict(subset="1"),
    ):
        with pytest.raises(ValueError):
            check(prepared, **changed)


def test_case_rejects_rehashed_treatment_and_heldout_geography_substitution(prepared):
    original = json.loads(prepared.read_text())
    changed = dict(original, treatment=dict(original["treatment"], ego_signal="Green"))
    prepared.write_text(json.dumps(changed))
    prepared.with_suffix(".sha256").write_text(digest(prepared))
    with pytest.raises(ValueError, match="treatment"):
        check(prepared)
    changed = dict(
        original, strict_allowed_junctions=original["strict_allowed_junctions"] + [999999]
    )
    prepared.write_text(json.dumps(changed))
    prepared.with_suffix(".sha256").write_text(digest(prepared))
    with pytest.raises(ValueError, match="reserved geography"):
        check(prepared)

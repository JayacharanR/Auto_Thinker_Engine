"""Pilot cannot draw calibration scenes or promote an observed diagnostic to TRAIN."""

import pytest

from scripts.collect_navigation_intervention_pilot import pilot_cases


def test_pilot_is_fixed_train_only_and_records_the_single_declared_repeat():
    train = [
        dict(context_id=f"r{route}_{kind}", route_id=route, mechanism=kind, role="train")
        for route, kind in ((35, "lead"), (134, "crossing"), (35, "rear"))
    ]
    calibration = [
        dict(context_id="calib_" + kind, route_id=1, mechanism=kind, role="calibration")
        for kind in ("lead", "crossing", "rear")
    ]
    result = pilot_cases(dict(contexts=list(reversed(train + calibration))))
    assert len(result) == 10 and all(not c["context_id"].startswith("calib") for c in result)
    assert sum(c["repeat"] == 1 for c in result) == 1 and result[-1]["branch"] == "go"
    bad = train + [dict(context_id="r6_lead", route_id=6, mechanism="lead", role="train")]
    with pytest.raises(ValueError, match="Old observed diagnostic"):
        pilot_cases(dict(contexts=bad))

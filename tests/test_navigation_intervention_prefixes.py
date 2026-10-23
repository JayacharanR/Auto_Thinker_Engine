import copy

import pytest

from scripts.audit_navigation_intervention_prefixes import compare_geometry_prefixes


def trace():
    actors = [
        dict(
            id=1,
            type="traffic.traffic_light",
            role="",
            pose=[0.0] * 6,
            velocity=[0.0] * 3,
            signal="Green",
            phase_elapsed=0.0,
        ),
        dict(
            id=2,
            type="traffic.traffic_light",
            role="",
            pose=[30.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            velocity=[0.0] * 3,
            signal="Green",
            phase_elapsed=0.0,
        ),
        dict(id=3, type="vehicle.test", role="hero", pose=[0.0] * 6, velocity=[0.0] * 3),
    ]
    return [dict(frame=i, actors=copy.deepcopy(actors)) for i in range(80)]


def test_order_and_episode_ids_do_not_define_static_correspondence():
    first, second = trace(), trace()
    for frame in second:
        frame["actors"] = list(reversed(frame["actors"]))
        for i, actor in enumerate(frame["actors"]):
            actor["id"] = i + 100
    result = compare_geometry_prefixes(first, second)
    assert result["observable_prefix_within_tolerance"]
    assert all(v == 0 for v in result["maximum_differences"].values())


def test_actual_motion_or_phase_difference_still_fails():
    first, second = trace(), trace()
    second[70]["actors"][2]["pose"][0] = 0.2
    assert not compare_geometry_prefixes(first, second)["observable_prefix_within_tolerance"]
    second = trace()
    second[60]["actors"][0]["phase_elapsed"] = 0.1
    assert not compare_geometry_prefixes(first, second)["observable_prefix_within_tolerance"]


def test_duplicate_geometry_is_rejected():
    first = trace()
    first[0]["actors"][1]["pose"] = [0.0] * 6
    with pytest.raises(ValueError, match="Ambiguous"):
        compare_geometry_prefixes(first, trace())

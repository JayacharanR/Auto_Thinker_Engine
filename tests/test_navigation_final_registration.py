"""Final route coverage is directional, censored near endpoints and geometry-only."""

from types import SimpleNamespace

import pytest

from scripts.register_navigation_final_suite import (
    match_explicit_stop_signals,
    registration_opportunities,
    trace_serialized_route,
)


def test_registration_counts_lane_specific_opportunities_and_excludes_endpoint_events():
    trace = []
    for x in range(101):
        wp = SimpleNamespace(
            road_id=1,
            section_id=0,
            lane_id=-1,
            transform=SimpleNamespace(location=SimpleNamespace(x=x, y=0, z=0)),
        )
        trace.append((wp, None))
    inventory = dict(
        stops=[
            dict(id=1, affected_lanes=[dict(lane=[1, 0, -1], xyz=[50, 0, 0])]),
            dict(id=2, affected_lanes=[dict(lane=[1, 0, 1], xyz=[50, 0, 0])]),
            dict(id=3, affected_lanes=[dict(lane=[1, 0, -1], xyz=[5, 0, 0])]),
            dict(id=4, affected_lanes=[dict(lane=[1, 0, -1], xyz=[95, 0, 0])]),
        ],
        lights=[dict(id=5, stop_lanes=[dict(lane=[1, 0, -1], xyz=[60, 0, 0])])],
    )
    found, arc = registration_opportunities(trace, inventory)
    assert arc[-1] == 100
    assert [e["actor_id"] for e in found["stop"]] == [1]
    assert [e["actor_id"] for e in found["signal"]] == [5]


def test_all_declared_waypoint_segments_are_interpolated_in_order():
    calls = []
    planner = SimpleNamespace(trace_route=lambda a, b: calls.append((a, b)) or [(a, b)])
    carla = SimpleNamespace(Location=lambda *xyz: xyz)
    positions = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (30, 10, 0)]
    assert len(trace_serialized_route(planner, carla, positions)) == 3
    assert calls == list(zip(positions, positions[1:]))


def test_five_explicit_signals_match_physical_runtime_actors_without_dropping_extras(tmp_path):
    xodr = tmp_path / "map.xodr"
    xodr.write_text(
        "<root><signals>"
        + "".join(f'<signal id="{i}" type="206" />' for i in range(5))
        + "</signals></root>"
    )
    landmarks = [
        SimpleNamespace(
            id=str(i), transform=SimpleNamespace(location=SimpleNamespace(x=10 * i, y=0, z=0))
        )
        for i in range(5)
    ]
    world_map = SimpleNamespace(get_all_landmarks_of_type=lambda _: landmarks)
    stops = [dict(id=100 + i, xyz=[10 * i, 0, 0]) for i in range(5)]
    stops.append(dict(id=200, xyz=[100, 0, 0]))
    matches = match_explicit_stop_signals(world_map, xodr, stops)
    assert [m["actor_id"] for m in matches] == list(range(100, 105))
    assert len(stops) == 6
    with pytest.raises(ValueError, match="unique"):
        match_explicit_stop_signals(world_map, xodr, stops + [dict(id=201, xyz=[0, 0, 0])])

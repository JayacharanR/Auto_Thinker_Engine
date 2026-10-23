import numpy as np
import pytest

from src.navigation.constant_velocity import PerceivedTrack, collision_risk, forecast
from src.navigation.trajectory_candidates import candidates, common_cost


def pool():
    navigation = np.column_stack([np.arange(2, 34, 2), np.zeros(16)])
    return candidates(navigation, np.zeros(8), np.full(8, 6.0), 4.0)


def test_all15_candidates_share_physical_limits_and_no_zero_distance_magic_brake():
    values = pool()
    assert len(values) == 15 and len({(c.path_index, c.speed_index) for c in values}) == 15
    for c in values:
        delta = np.diff(np.r_[4.0, c.speeds])
        assert delta.min() >= -1.75 and delta.max() <= 1.25
        assert c.waypoints.shape == (8, 2) and (c.speeds >= 0).all()
    stop = values[0]
    assert stop.waypoints[0, 0] > 0 and stop.speeds[0] > 0
    waiting = values[4]
    stopped = np.flatnonzero(waiting.speeds < 0.1)
    assert len(stopped) >= 2 and waiting.speeds[stopped[-1] + 1] > 0


def test_cv_forecast_tracks_crossing_motion_and_uncertainty_does_not_lower_risk():
    values = pool()
    cruise = next(c for c in values if c.path_index == 1 and c.speed_index == 2)
    crossing = PerceivedTrack(1, np.array([10.0, -6.0]), np.array([0.0, 4.0]))
    points, sigma = forecast(crossing, cruise.times)
    assert points[2, 1] == 0 and np.all(np.diff(sigma) > 0)
    assert collision_risk(cruise, [crossing]).max() > 0.8
    far = PerceivedTrack(2, np.array([60.0, 20.0]), np.zeros(2), velocity_sigma_mps=0)
    assert collision_risk(cruise, [far]).max() < 0.001
    with pytest.raises(ValueError):
        forecast(PerceivedTrack(3, np.array([np.nan, 0]), np.zeros(2)), cruise.times)


def test_shared_cost_can_prefer_safe_stop_over_progress_but_remains_predictor_agnostic():
    values = pool()
    stop = values[0]
    go = values[2]
    assert common_cost(stop, np.zeros(8), np.zeros(8)) < common_cost(go, np.ones(8), np.zeros(8))
    assert common_cost(go, np.zeros(8), np.zeros(8)) < common_cost(stop, np.zeros(8), np.zeros(8))


def test_lateral_wiggles_do_not_create_ordered_route_progress():
    from src.navigation.trajectory_candidates import ordered_projection

    route = np.column_stack([np.arange(2, 34, 2), np.zeros(16)])
    xy = np.column_stack([np.full(8, 10.0), np.array([-2, 2, -2, 2, -2, 2, -2, 2])])
    projected = ordered_projection(xy, route)
    assert np.allclose(projected, 10.0)


def test_measured_overspeed_cannot_teleport_to_the_desired_cap():
    route = np.column_stack([np.arange(2, 34, 2), np.zeros(16)])
    for candidate in candidates(route, np.zeros(8), np.full(8, 8.0), 12.0):
        assert np.diff(np.r_[12, candidate.speeds]).min() >= -1.75
        assert candidate.speeds[0] == 10.25

"""Action-intervention paths must cross physically, rather than at an overpass."""

import numpy as np

from scripts.prepare_navigation_intervention_geometry import first_intersection


def test_crossing_is_perpendicular_and_at_the_same_elevation():
    ego = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    other = np.array([[20.0, -10.0, 0.0], [20.0, 20.0, 0.0]])
    assert np.allclose(first_intersection(ego, other), (20.0, 10.0))
    overpass = other.copy()
    overpass[:, 2] = 5
    assert first_intersection(ego, overpass) is None
    parallel = np.array([[0.0, 1.0, 0.0], [30.0, 1.0, 0.0]])
    assert first_intersection(ego, parallel) is None

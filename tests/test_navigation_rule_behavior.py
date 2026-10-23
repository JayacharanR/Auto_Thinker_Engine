from copy import deepcopy

from src.navigation.rule_behavior import behavior


def trace(seconds=8.0, speed=0.0, distance=3.0, light="Red"):
    return [
        dict(
            scripted_setup=False,
            timestamp=i * 0.05,
            speed_mps=speed,
            distance_to_registered_line_m=distance,
            ego=[0.0, 0.0, 0.0, 0.0],
            observed_light=light,
        )
        for i in range(round(seconds * 20) + 1)
    ]


def test_stationary_red_memory_cannot_pass_without_setup_certificate_or_full_interval():
    spec = dict(treatment=dict(family="out_of_view_memory", arm=0))
    native = dict(red_violations=0, stop_violations=0, completed=False)
    contacts = dict(complete=True, ego_contact_starts=0)
    raw = behavior(spec, trace(), native, contacts)
    assert raw["behavior_condition_satisfied"] and not raw["passed"]
    assert behavior(spec, trace(), native, contacts, True)["passed"]
    assert not behavior(spec, trace(7.9), native, contacts, True)["passed"]
    assert not behavior(spec, trace(), native, dict(contacts, ego_contact_starts=1), True)["passed"]
    assert not behavior(spec, trace(), native, dict(contacts, complete=False), True)["passed"]


def test_green_memory_requires_departure_and_close_yellow_retains_safe_stop_failure():
    native = dict(red_violations=0, stop_violations=0, completed=False)
    contacts = dict(complete=True, ego_contact_starts=0)
    held = trace(light="Green")
    green = dict(treatment=dict(family="out_of_view_memory", arm=1))
    assert not behavior(green, held, native, contacts, True)["passed"]
    moving = deepcopy(held)
    for row in moving:
        row["ego"][0] = row["timestamp"] * 1.1
    assert behavior(green, moving, native, contacts, True)["passed"]
    close = dict(treatment=dict(family="yellow_distance", arm=0))
    # Predeclared close-amber objective is to clear. A collision-free stop is
    # retained as failure of that objective, with no claim that it is illegal.
    result = behavior(close, trace(seconds=12, light="Yellow"), native, contacts, True)
    assert result["common_safety_condition"] and not result["passed"]

"""A larger queue cannot turn failed pilot options into favorable retries."""

from scripts.collect_navigation_intervention_expansion import remaining_cases


def test_remaining_cases_exclude_all_pilot_options_not_only_successful_captures():
    contexts = []
    for route in (35, 50, 134, 189, 225, 231, 258, 297):
        for mechanism in ("lead", "rear") + (("crossing",) if route in (134, 231) else ()):
            contexts.append(
                dict(
                    context_id=f"r{route}_{mechanism}", role="train", town="TownXX", route_id=route
                )
            )
    for route in (124, 211):
        for mechanism in ("lead", "rear", "crossing"):
            contexts.append(
                dict(
                    context_id=f"r{route}_{mechanism}",
                    role="calibration",
                    town="TownXX",
                    route_id=route,
                )
            )
    pilot = dict(cases=[dict(context_id=c) for c in ("r35_lead", "r134_crossing", "r35_rear")])
    cases = remaining_cases(dict(contexts=contexts), pilot)
    ids = {c["case_id"] for c in cases}
    assert len(ids) == 63 and sum(c["role"] == "calibration" for c in cases) == 18
    assert "r35_lead_brake_repeat0" not in ids  # Failed before driving, still not a retry.
    assert "r134_crossing_wait_go_repeat0" not in ids
    assert "r50_lead_brake_repeat0" in ids

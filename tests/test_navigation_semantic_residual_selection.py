from copy import deepcopy

from scripts.run_navigation_semantic_study import admitted


def scores():
    return dict(counts=dict(stopped=560, controller_go=582), controller_stop_recall=.55,
                controller_false_hold_fraction=.139, go_speed_mae_mps=4.59,
                selection_score=.386, families=dict(signal=dict(stopped=100, stop_hits=60)))


def test_aggregate_gain_cannot_hide_lost_signal_behavior():
    parent = scores()
    candidate = dict(deepcopy(parent), controller_stop_recall=.60, selection_score=.35)
    assert admitted(candidate, parent)
    candidate["families"]["signal"]["stop_hits"] = 57
    assert not admitted(candidate, parent)


def test_stopping_everywhere_and_small_changes_cannot_pass_protection():
    parent = scores()
    candidate = dict(deepcopy(parent), controller_stop_recall=.90, selection_score=.35,
                     controller_false_hold_fraction=.20)
    assert not admitted(candidate, parent)
    candidate = dict(deepcopy(parent), controller_stop_recall=.56, selection_score=.38)
    assert not admitted(candidate, parent)

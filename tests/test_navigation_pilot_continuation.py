import pytest

from scripts.assess_navigation_student_pilot import episode_evidence_issue
from scripts.continue_navigation_corrected_pilots import unstarted_candidates


def test_first_response_crash_is_incomplete_evidence(tmp_path):
    # A real failure mode: empty traces but no profile/video, no policy traceback.
    for name in ("trace.jsonl", "world_trace.jsonl", "policy_stderr.log"):
        (tmp_path / name).touch()
    assert episode_evidence_issue(tmp_path) == dict(
        missing=["profile.json", "three_cameras.mp4"],
        empty=["trace.jsonl", "world_trace.jsonl"],
    )


def test_settled_failure_is_never_retried():
    candidates = [dict(name=str(i), sha256=str(i)) for i in range(5)]
    failed = dict(candidate=candidates[0], settled_routes=6, exit_code=1)
    assert unstarted_candidates(dict(candidates=candidates), [failed]) == candidates[1:]
    with pytest.raises(ValueError, match="incomplete"):
        unstarted_candidates(dict(candidates=candidates), [dict(failed, settled_routes=5)])
    with pytest.raises(ValueError, match="exact prefix"):
        unstarted_candidates(dict(candidates=candidates), [dict(failed, candidate=candidates[1])])
    censored = dict(failed, settled_routes=5, planned_wall_budget_verified=True,
                    unknown_failure_routes=1)
    assert unstarted_candidates(dict(candidates=candidates), [censored]) == candidates[1:]
    with pytest.raises(ValueError, match="incomplete"):
        unstarted_candidates(
            dict(candidates=candidates), [dict(censored, unknown_failure_routes=0)]
        )

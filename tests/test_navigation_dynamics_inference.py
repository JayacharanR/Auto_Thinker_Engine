import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.navigation.dynamics_inference import (
    INFERENCE_SOURCES,
    FactualDynamicsEnsemble,
    file_digest,
)
from src.navigation.ordered_dynamics import OrderedJEPADynamics
from src.navigation.rssm_dynamics import RSSMDynamics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=("jepa", "rssm"))
def ensemble(tmp_path, request):
    torch.set_num_threads(2)
    kind = request.param
    members = []
    # Synthetic untrained unit fixtures, never scientific export artifacts.
    with torch.random.fork_rng(devices=[]):
        for seed in (42, 1051, 2060):
            torch.manual_seed(seed)
            model = (OrderedJEPADynamics if kind == "jepa" else RSSMDynamics)(spatial_tokens=48)
            path = tmp_path / f"{kind}_{seed}.pt"
            torch.save(dict(model=model.state_dict(), hidden=128, spatial_tokens=48,
                            architecture="ordered-jepa" if kind == "jepa" else "rssm", step=0,
                            request_sha256="synthetic unit fixture only",
                            red_risk_qualified=False, planner_qualified=False), path)
            members.append(dict(seed=seed, checkpoint=str(path), sha256=file_digest(path),
                                selected_step=0, request_sha256="synthetic unit fixture only"))
    descriptor = dict(schema="navigation-factual-ensemble-v1", model=kind, members=members,
                      encoder_descriptor_sha256="fixed_encoder", red_risk_qualified=False,
                      planner_qualified=False, calibration_qualified=False,
                      inference_source_sha256={p: file_digest(ROOT / p) for p in INFERENCE_SOURCES})
    return FactualDynamicsEnsemble(descriptor, device="cpu"), descriptor


def inputs():
    rng = np.random.default_rng(23)
    tokens = rng.normal(size=(48, 384)).astype(np.float32)
    actions = np.zeros((2, 80, 3), np.float32)
    actions[..., 1] = .5
    nav = np.column_stack([np.arange(2, 34, 2), np.zeros(16)]).astype(np.float32)
    return dict(past_tokens=tokens, ordered_controls=actions, navigation=nav,
                ego=np.array([4., .7, .01], np.float32), encoder_sha256="fixed_encoder")


def test_inference_matches_training_conditioning_and_omits_untrained_red(ensemble):
    model, _ = ensemble
    values = inputs()
    result = model.predict(**values)
    condition = np.r_[values["navigation"].ravel() / 32, values["ego"] / [10, np.pi, 1]]
    with torch.inference_mode():
        direct = model.members[0](
            torch.from_numpy(values["past_tokens"])[None].expand(2, -1, -1),
            torch.from_numpy(values["ordered_controls"]),
            conditioning=torch.tensor(condition, dtype=torch.float32)[None].expand(2, -1),
        )["outcome_logits"].float()
    assert np.allclose(result["collision_members"][0], direct[..., 0].sigmoid().numpy())
    assert np.allclose(result["progress_members_m"][0], direct[..., 2].numpy())
    assert result["red_prediction"] is None and result["untrained_red_row_ignored"]
    assert not result["planner_qualified"] and not result["calibration_qualified"]


def test_late_actions_cannot_affect_earlier_horizons_or_another_candidate(ensemble):
    model, _ = ensemble
    values = inputs()
    first = model.predict(**values)
    values["ordered_controls"][1, 60:, 1] = 0
    values["ordered_controls"][1, 60:, 2] = 1
    second = model.predict(**values)
    assert np.array_equal(first["collision_mean"][0], second["collision_mean"][0])
    assert np.array_equal(first["collision_mean"][1, :6], second["collision_mean"][1, :6])
    assert not np.allclose(first["progress_mean_m"][1, 6:], second["progress_mean_m"][1, 6:])
    with pytest.raises(TypeError):
        model.predict(**values, future_tokens=np.zeros((8, 48, 384)))
    with pytest.raises(ValueError, match="identical frozen"):
        model.predict(**dict(values, encoder_sha256="wrong_encoder"))


def test_false_qualification_or_changed_member_source_is_rejected(ensemble):
    _, descriptor = ensemble
    changed = deepcopy(descriptor)
    changed["planner_qualified"] = True
    with pytest.raises(ValueError, match="explicitly unqualified"):
        FactualDynamicsEnsemble(changed, device="cpu")
    changed = deepcopy(descriptor)
    changed["inference_source_sha256"][INFERENCE_SOURCES[0]] = "wrong"
    with pytest.raises(ValueError, match="inference equations changed"):
        FactualDynamicsEnsemble(changed, device="cpu")


def test_fresh_policy_import_cannot_access_training_or_simulator_modules():
    script = (
        "import sys\n"
        "from src.navigation.policy_worker import PrivilegeGuard, FORBIDDEN\n"
        "sys.meta_path.insert(0, PrivilegeGuard())\n"
        "from src.navigation.dynamics_inference import FactualDynamicsEnsemble\n"
        "assert not any(name in sys.modules for name in FORBIDDEN)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr

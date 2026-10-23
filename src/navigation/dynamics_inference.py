"""Past-only factual ensemble inference; raw predictions do not qualify a planner."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.navigation.ordered_dynamics import OrderedJEPADynamics
from src.navigation.rssm_dynamics import RSSMDynamics

INFERENCE_SOURCES = (
    "src/navigation/dynamics_inference.py",
    "src/navigation/ordered_dynamics.py",
    "src/navigation/rssm_dynamics.py",
)


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class FactualDynamicsEnsemble:
    """All three independent members; omit the untrained red prediction.

    Only current RGB tokens, declared route/ego values and proposed ordered
    controls enter forward. Targets, labels, world traces and actor inventories
    cannot be supplied. No posterior or stochastic inference is used for RSSM.
    """

    def __init__(self, descriptor, device="cuda"):
        if isinstance(descriptor, (str, Path)):
            descriptor = json.loads(Path(descriptor).read_text())
        if (
            descriptor.get("schema") != "navigation-factual-ensemble-v1"
            or descriptor["model"] not in ("jepa", "rssm")
            or descriptor.get("red_risk_qualified") is not False
            or descriptor.get("planner_qualified") is not False
            or descriptor.get("calibration_qualified") is not False
            or len(descriptor["members"]) != 3
            or sorted(m["seed"] for m in descriptor["members"]) != [42, 1051, 2060]
        ):
            raise ValueError("Use the exact three-member, explicitly unqualified factual export")
        sources = descriptor["inference_source_sha256"]
        root = Path(__file__).resolve().parents[2]
        if set(sources) != set(INFERENCE_SOURCES) or any(
            file_digest(root / path) != expected for path, expected in sources.items()
        ):
            raise ValueError("Source-frozen inference equations changed")
        self.device = torch.device(device)
        self.descriptor = dict(descriptor)
        self.members = []
        expected_arch = "ordered-jepa" if descriptor["model"] == "jepa" else "rssm"
        for member in descriptor["members"]:
            if file_digest(member["checkpoint"]) != member["sha256"]:
                raise ValueError("Frozen dynamics member bytes changed")
            state = torch.load(member["checkpoint"], map_location="cpu", weights_only=True)
            if (
                state["architecture"] != expected_arch or state["spatial_tokens"] != 48
                or state["hidden"] != 128 or state["step"] != member["selected_step"]
                or state["request_sha256"] != member["request_sha256"]
                or state["red_risk_qualified"] is not False
                or state["planner_qualified"] is not False
            ):
                raise ValueError("Member checkpoint does not bind the declared factual model")
            model = (OrderedJEPADynamics if descriptor["model"] == "jepa" else RSSMDynamics)(
                spatial_tokens=48, hidden=128)
            model.load_state_dict(state["model"], strict=True)
            self.members.append(model.to(self.device).eval().requires_grad_(False))

    @torch.inference_mode()
    def predict(self, past_tokens, ordered_controls, navigation, ego, encoder_sha256):
        if encoder_sha256 != self.descriptor["encoder_descriptor_sha256"]:
            raise ValueError("Current tokens must use the identical frozen factual encoder")
        tokens = torch.as_tensor(past_tokens, device=self.device).float()
        actions = torch.as_tensor(ordered_controls, device=self.device).float()
        nav = torch.as_tensor(navigation, device=self.device).float()
        motion = torch.as_tensor(ego, device=self.device).float()
        if (
            tokens.shape != (48, 384) or nav.shape != (16, 2) or motion.shape != (3,)
            or actions.ndim != 3 or actions.shape[1:] != (80, 3)
            or not 1 <= len(actions) <= 32
            or not all(torch.isfinite(x).all() for x in (tokens, actions, nav, motion))
            or (actions[..., 0].abs() > 1).any()
            or (actions[..., 1:] < 0).any() or (actions[..., 1:] > 1).any()
        ):
            raise ValueError(
                "Past48-token grid, current nav/ego and bounded80-control plans required"
            )
        condition = torch.cat([
            nav.flatten() / 32,
            motion / motion.new_tensor([10, np.pi, 1]),
        ])[None].expand(len(actions), -1)
        current = tokens[None].expand(len(actions), -1, -1)
        predictions = []
        with torch.autocast(self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            for model in self.members:
                prediction = model(current, actions, conditioning=condition)
                predictions.append(prediction["outcome_logits"].float())
        values = torch.stack(predictions)
        if not torch.isfinite(values[..., [0, 2]]).all():
            raise FloatingPointError("Nonfinite factual ensemble predictions")
        collision = values[..., 0].sigmoid()
        progress = values[..., 2]
        return dict(
            collision_mean=collision.mean(0).cpu().numpy(),
            collision_members=collision.cpu().numpy(),
            collision_member_variance_uncalibrated=collision.var(0, correction=0).cpu().numpy(),
            progress_mean_m=progress.mean(0).cpu().numpy(),
            progress_members_m=progress.cpu().numpy(),
            red_prediction=None, untrained_red_row_ignored=True,
            planner_qualified=False, calibration_qualified=False,
            limitations="Collision-balanced training; raw probabilities and member variance "
            "are not calibrated. No positive red supervision in this generation.",
        )

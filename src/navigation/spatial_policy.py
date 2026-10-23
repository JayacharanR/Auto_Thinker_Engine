"""Three-camera spatial policy: RGB-only feature extraction and causal control."""

from collections import deque

import numpy as np
import torch
from PIL import Image

from src.navigation.contracts import VIEWS
from src.navigation.controller import track, trajectory
from src.navigation.grid_encoders import FrozenGridEncoder
from src.navigation.optical_fields import optical_fields
from src.navigation.spatial_student import SpatialCameraStudent


class SpatialPolicy:
    def __init__(self, state, device="cuda", rule_overrides=False):
        if rule_overrides:
            raise ValueError("Spatial policy has no trained auxiliary rule-override layer")
        self.device = torch.device(device)
        self.kind = state["encoder_kind"]
        model_type = SpatialCameraStudent
        if state["architecture"] == "spatial-visual-decision-v2":
            from src.navigation.visual_decision_student import VisualDecisionStudent

            model_type = VisualDecisionStudent
        self.model = (
            model_type(
                state["encoder_dim"],
                state["hidden"],
                scratch=self.kind == "cnn_scratch",
                color_enabled=state["color_enabled"],
            )
            .to(self.device)
            .eval()
        )
        self.model.load_state_dict(state["model"], strict=True)
        self.encoder = (
            None
            if self.kind in ("cnn_scratch", "nav_color_only")
            else FrozenGridEncoder(state["encoder_descriptor"]).to(self.device)
        )
        self.context, self.features = deque(maxlen=8), deque(maxlen=8)
        self.previous, self.hidden, self.prediction, self.plan = None, None, None, None
        self.ticks = 0

    @torch.inference_mode()
    def step(self, packet):
        packet.validate()
        if packet.navigation_waypoints.shape != (16, 2):
            raise ValueError("Sixteen declared navigation points required")
        if self.previous and (
            packet.frame != self.previous[0] + 1
            or abs(packet.timestamp - self.previous[1] - 0.05) > 1e-5
        ):
            raise ValueError("Spatial policy requires contiguous20Hz sensor packets")
        self.previous = packet.frame, packet.timestamp
        if self.ticks % 2 == 0:
            down = np.stack(
                [
                    np.asarray(
                        Image.fromarray(packet.rgb[v]).resize((256, 144), Image.Resampling.BILINEAR)
                    ).transpose(2, 0, 1)
                    for v in VIEWS
                ]
            )
            rgb = torch.from_numpy(down.copy()).to(self.device)
            if self.kind in ("jepa_comma2k19", "jepa_carla", "jepa_random"):
                if not self.context:
                    self.context.extend([rgb] * 8)
                else:
                    self.context.append(rgb)
                clip = torch.stack(list(self.context), 1)[None]
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    visual = self.encoder.video_features(clip).half().float()
            elif self.kind == "nav_color_only":
                visual = torch.zeros(1, 3, 16, 384, device=self.device)
            else:
                if self.kind == "cnn_scratch":
                    current = self.model.scratch(rgb).float()
                else:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda",
                    ):
                        current = self.encoder.frame_features(rgb).float()
                if not self.features:
                    self.features.extend([current] * 8)
                else:
                    self.features.append(current)
                visual = torch.stack(list(self.features), 0).mean(0)[None]
                if self.kind != "cnn_scratch":
                    visual = visual.half().float()
            native = torch.from_numpy(
                np.stack([packet.rgb[v].transpose(2, 0, 1) for v in VIEWS]).copy()
            ).to(self.device)
            fields = optical_fields(native).half().float()
            inputs = dict(
                features=visual[:, None],
                optical=fields[None, None],
                navigation=torch.from_numpy(packet.navigation_waypoints.copy()).to(self.device)[
                    None, None
                ],
                ego=torch.from_numpy(packet.ego_motion.copy()).to(self.device)[None, None],
            )
            p, self.hidden = self.model(**inputs, hidden=self.hidden)
            self.prediction = {k: v[0, 0].float().cpu().numpy() for k, v in p.items()}
        p = self.prediction
        if self.ticks % 4 == 0:
            self.plan = trajectory(packet, p["correction"], p["speeds"], np.zeros(4))
        brake = float(torch.sigmoid(torch.as_tensor(p["brake"])))
        command = track(
            packet.navigation_waypoints,
            p["correction"],
            float(packet.ego_motion[0]),
            float(p["current_speed"]),
            brake,
        )
        self.ticks += 1
        return dict(
            frame=packet.frame,
            control=[command.steer, command.throttle, command.brake],
            target_speed_mps=float(p["current_speed"]),
            brake_probability=brake,
            light_probabilities=None,
            predicted_distances_m=None,
            auxiliary_perception_available=False,
            plan_waypoints=self.plan.waypoints.tolist(),
            plan_speeds=self.plan.speeds.tolist(),
            rule_overrides_enabled=False,
            interventions=[],
        )

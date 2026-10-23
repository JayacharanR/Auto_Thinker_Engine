"""Inference uses only approved packet arrays and learned perception predictions."""

import numpy as np
import torch
from PIL import Image

from src.navigation.contracts import VIEWS
from src.navigation.controller import track, trajectory
from src.navigation.student import make_student


class CameraPolicy:
    def __init__(self, checkpoint, device="cuda", rule_overrides=False):
        torch.set_num_threads(2)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["architecture"] in ("spatial-stop-go-v1", "spatial-visual-decision-v2"):
            from src.navigation.spatial_policy import SpatialPolicy

            self.spatial_policy = SpatialPolicy(state, device, rule_overrides)
            return
        self.device = torch.device(device)
        self.model = make_student(state["architecture"], state["hidden"]).to(self.device).eval()
        self.model.load_state_dict(state["model"])
        self.signal_facts = None
        self.fact_arm = state.get("fact_arm")
        if state["architecture"] == "camera-signal-residual-v3":
            import hashlib
            from pathlib import Path

            from src.navigation.signal_perception import FrozenSignalPerception
            from src.navigation.signal_policy import CausalSignalFacts

            bundle = Path(state["signal_bundle"])
            if hashlib.sha256(bundle.read_bytes()).hexdigest() != state["signal_bundle_sha256"]:
                raise ValueError("Driving checkpoint's declared perception bundle changed")
            if self.fact_arm not in ("rgb", "null"):
                raise ValueError("Driving checkpoint must declare RGB/null fact arm")
            if self.fact_arm == "rgb":
                self.signal_facts = CausalSignalFacts(FrozenSignalPerception(bundle, device))
        self.rule_overrides = rule_overrides
        self.hidden, self.previous, self.prediction, self.plan = None, None, None, None
        self.ticks = 0
        self.red_memory = False
        self.stop_since, self.stop_released = None, False

    @torch.no_grad()
    def step(self, packet):
        if hasattr(self, "spatial_policy"):
            return self.spatial_policy.step(packet)
        packet.validate()
        if packet.navigation_waypoints.shape != (16, 2):
            raise ValueError("Camera student expects sixteen navigation points")
        if self.previous and (
            packet.frame != self.previous[0] + 1
            or abs(packet.timestamp - self.previous[1] - 0.05) > 1e-5
        ):
            raise ValueError("Camera policy requires contiguous 20Hz packets")
        self.previous = packet.frame, packet.timestamp
        if self.ticks % 2 == 0:  # Network memory consumes aligned history at 10Hz.
            images = np.stack(
                [
                    np.asarray(
                        Image.fromarray(packet.rgb[v]).resize((256, 144), Image.Resampling.BILINEAR)
                    ).transpose(2, 0, 1)
                    for v in VIEWS
                ]
            )
            tensors = {
                "rgb": images,
                "front": packet.rgb["front"].transpose(2, 0, 1),
                "navigation": packet.navigation_waypoints.astype(np.float32),
                "ego": packet.ego_motion.astype(np.float32),
            }
            inputs = {
                k: torch.from_numpy(np.ascontiguousarray(v)).to(self.device)[None, None]
                for k, v in tensors.items()
            }
            if self.fact_arm is not None:
                from src.navigation.signal_policy import MAX_OBJECTS, TOKEN_DIM

                if self.fact_arm == "rgb":
                    facts, valid = self.signal_facts.step(packet)
                else:
                    facts = np.zeros((MAX_OBJECTS + 1, TOKEN_DIM), np.float32)
                    valid = np.zeros(MAX_OBJECTS + 1, bool)
                    valid[0] = True
                inputs.update(
                    signal_facts=torch.from_numpy(facts).to(self.device)[None, None],
                    signal_valid=torch.from_numpy(valid).to(self.device)[None, None],
                )
            prediction, self.hidden = self.model(**inputs, hidden=self.hidden)
            self.prediction = {k: v[0, 0].float().cpu().numpy() for k, v in prediction.items()}
        p = self.prediction
        if self.ticks % 4 == 0:  # A new four-second trajectory at 5Hz.
            self.plan = trajectory(
                packet,
                p["correction"],
                p["speeds"],
                torch.sigmoid(torch.from_numpy(p["hazards"])).numpy(),
            )
        target = float(p["current_speed"])
        brake_probability = float(torch.sigmoid(torch.as_tensor(p["brake"])))
        interventions = []
        if self.rule_overrides:
            light = torch.softmax(torch.from_numpy(p["light"]), -1).numpy()
            present = torch.sigmoid(torch.from_numpy(p["presence"])).numpy()
            distance, stop_distance = p["distance"]
            if light[1] > 0.8 and present[0] > 0.8 and distance < 12:
                self.red_memory = True
            if light[3] > 0.8:
                self.red_memory = False
            if self.red_memory:
                target = min(target, float(np.sqrt(2 * 2.5 * max(0, distance - 3))))
                if distance < 4:
                    brake_probability = 1
                interventions.append("perceived_red_light")
            if present[1] < 0.2:
                self.stop_released, self.stop_since = False, None
            if present[1] > 0.8 and stop_distance < 6 and not self.stop_released:
                target = min(target, float(np.sqrt(2 * 2.5 * max(0, stop_distance - 2))))
                if stop_distance < 3:
                    brake_probability = 1
                    if packet.ego_motion[0] < 0.1:
                        self.stop_since = self.stop_since or packet.timestamp
                        if packet.timestamp - self.stop_since >= 1:
                            self.stop_released = True
                interventions.append("perceived_stop_sign")
        command = track(
            packet.navigation_waypoints,
            p["correction"],
            float(packet.ego_motion[0]),
            target,
            brake_probability,
        )
        self.ticks += 1
        return {
            "frame": packet.frame,
            "control": [command.steer, command.throttle, command.brake],
            "target_speed_mps": target,
            "brake_probability": brake_probability,
            "light_probabilities": torch.softmax(torch.from_numpy(p["light"]), -1).tolist(),
            "predicted_distances_m": p["distance"].tolist(),
            "plan_waypoints": self.plan.waypoints.tolist(),
            "plan_speeds": self.plan.speeds.tolist(),
            "rule_overrides_enabled": self.rule_overrides,
            "interventions": interventions,
        }

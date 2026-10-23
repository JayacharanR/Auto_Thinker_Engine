"""Camera-fact deployment uses the same past-only spatial policy, overrides off."""

import torch

from src.navigation.fact_student import FactStudent
from src.navigation.spatial_policy import SpatialPolicy


class FactSpatialPolicy(SpatialPolicy):
    def __init__(self, state, device="cuda", rule_overrides=False):
        torch.set_num_threads(2)
        if state["architecture"] != "spatial-visual-facts-v3":
            raise ValueError("Explicit camera-fact architecture required")
        plain = dict(
            state,
            architecture="spatial-visual-decision-v2",
            model={k: v for k, v in state["model"].items() if not k.startswith("facts.")},
        )
        super().__init__(plain, device, rule_overrides)
        self.model = (
            FactStudent(
                state["encoder_dim"],
                state["hidden"],
                scratch=self.kind == "cnn_scratch",
                color_enabled=state["color_enabled"],
            )
            .to(self.device)
            .eval()
        )
        self.model.load_state_dict(state["model"], strict=True)

    @torch.inference_mode()
    def step(self, packet):
        result = super().step(packet)
        prediction = self.prediction
        result.update(
            light_probabilities=torch.softmax(torch.as_tensor(prediction["light"]), -1).tolist(),
            predicted_distances_m=prediction["distance"].tolist(),
            presence_probabilities=torch.sigmoid(torch.as_tensor(prediction["presence"])).tolist(),
            hazard_probabilities=torch.sigmoid(torch.as_tensor(prediction["hazards"])).tolist(),
            auxiliary_perception_available=True,
            auxiliary_predictions_used_for_control=False,
        )
        return result

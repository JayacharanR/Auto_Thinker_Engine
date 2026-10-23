"""Spatial camera decision model without an instantaneous ego-speed shortcut."""

from src.navigation.spatial_student import SpatialCameraStudent


class VisualDecisionStudent(SpatialCameraStudent):
    def forward(self, features, optical, navigation, ego, hidden=None):
        # The physical controller still receives the measured speed. This is a
        # declared neural-input ablation, never a change to the sensor packet,
        # teacher targets, scoring rules, or actuator state.
        decision_ego = ego.clone()
        decision_ego[..., 0] = 0
        return super().forward(features, optical, navigation, decision_ego, hidden)

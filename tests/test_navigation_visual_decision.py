import numpy as np
import torch

from src.navigation.controller import track
from src.navigation.visual_decision_student import VisualDecisionStudent


def test_neural_speed_ablation_preserves_packet_inputs_and_controller_speed():
    torch.set_num_threads(2)
    model = VisualDecisionStudent(encoder_dim=24, hidden=32).eval()
    inputs = dict(
        features=torch.randn(1, 5, 3, 16, 24),
        optical=torch.randn(1, 5, 3, 144, 11),
        navigation=torch.randn(1, 5, 16, 2),
        ego=torch.randn(1, 5, 3),
    )
    original = inputs["ego"].clone()
    with torch.inference_mode():
        a, hidden_a = model(**inputs)
        changed = dict(inputs, ego=original.clone())
        changed["ego"][..., 0] = 12
        b, hidden_b = model(**changed)
    assert torch.equal(inputs["ego"], original)
    assert all(torch.equal(value, b[key]) for key, value in a.items())
    assert torch.equal(hidden_a, hidden_b)
    nav = np.column_stack([np.arange(2, 34, 2), np.zeros(16)])
    slow = track(nav, np.zeros(8), 0, 4, 0)
    fast = track(nav, np.zeros(8), 7, 4, 0)
    assert slow.throttle > 0 and slow.brake == 0
    assert fast.brake > 0 and fast.throttle == 0

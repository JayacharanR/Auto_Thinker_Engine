import numpy as np

from src.navigation.dynamics_data import factual_window


def test_partial_control_tail_and_assistance_never_create_safe_future_labels():
    frames = np.arange(100, 202, 2)
    navigation = np.tile(np.c_[np.arange(2, 34, 2), np.zeros(16)], (51, 1, 1))
    ego = np.zeros((51, 3))
    positions = np.c_[np.arange(51) * 0.2, np.zeros((51, 2))]
    aligned = {f: dict(control=[0.1, 0.2, 0]) for f in range(101, 126)}
    row = factual_window(
        0,
        frames,
        navigation,
        ego,
        positions,
        aligned,
        [119],
        first_safety_frame=115,
        native_red_zero_observed=True,
    )
    assert row["known_control_ticks"] == 25
    assert row["latent_valid"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert row["outcomes"][0, 0] == 0 and row["outcomes"][1, 0] == 1
    assert row["cost_valid"][1, 1] == 0 and np.isnan(row["outcomes"][1, 1])
    assert (row["cost_valid"][2:] == 0).all()
    assert np.isnan(row["outcomes"][2:]).all()
    assert (row["controls"][25:] == 0).all()
    assert row["conditioning"].shape == (35,)


def test_unobserved_red_state_never_becomes_a_safe_red_target():
    frames = np.arange(100, 202, 2)
    navigation = np.tile(np.c_[np.arange(2, 34, 2), np.zeros(16)], (51, 1, 1))
    ego = np.zeros((51, 3))
    positions = np.c_[np.arange(51), np.zeros((51, 2))]
    aligned = {f: dict(control=[0, 0.2, 0]) for f in range(101, 201)}
    row = factual_window(
        0,
        frames,
        navigation,
        ego,
        positions,
        aligned,
        [],
        first_safety_frame=115,
        native_red_zero_observed=False,
    )
    assert (row["cost_valid"][:, 1] == 0).all()
    assert (row["cost_valid"][1:, 0] == 0).all()
    assert (row["cost_valid"][:, 2] == 1).all()

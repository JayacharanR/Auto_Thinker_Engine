"""Training-only event targets; no simulator observations enter policy inputs."""

import numpy as np


def first_contact_targets(known_ticks, first_contact_tick):
    if type(known_ticks) is not int or not 0 <= known_ticks <= 80:
        raise ValueError("Explicit0..80observed prefix required")
    if first_contact_tick is not None and type(first_contact_tick) is not int:
        raise ValueError("Integer contact offset or unknown required")
    event, valid = np.zeros(80, np.float32), np.zeros(80, bool)
    if first_contact_tick is not None and first_contact_tick <= 0:
        return event, valid
    end = min(known_ticks, first_contact_tick or 80)
    valid[:end] = True
    if first_contact_tick is not None and first_contact_tick <= known_ticks:
        event[first_contact_tick - 1] = 1
    return event, valid


def latent_target_mask(known_ticks, first_contact_tick):
    # RGB exists only through physics tick158; branch is80, so+80is missing.
    steps = np.arange(1, 9) * 10
    valid = (steps <= known_ticks) & (steps < 80)
    if first_contact_tick is not None:
        valid &= steps < first_contact_tick
    return valid

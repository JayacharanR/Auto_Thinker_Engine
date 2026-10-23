"""Offline paired-phase supervision; never imported by a driving policy."""

import numpy as np


def active_bulb_masks(red, green, yellow, red_repeat):
    """Return conservative pixel masks using fixed response/noise thresholds.

    Images are aligned, static native RGB crops of one previously verified
    responsive housing. No class is inferred from hue: phase changes identify
    the emissive pixels. Missing response means unknown, not background.
    """
    images = [np.asarray(x) for x in (red, green, yellow, red_repeat)]
    if any(x.dtype != np.uint8 or x.ndim != 3 or x.shape[-1] != 3 for x in images):
        raise ValueError("Four native uint8 HWC RGB crops required")
    if not images[0].size or any(x.shape != images[0].shape for x in images):
        raise ValueError("Paired crops must have identical nonempty shapes")
    red, green, yellow, repeat = [x.astype(np.float32) for x in images]
    noise = np.abs(red - repeat).mean(-1)
    # The lesser repeated red intensity must exceed both other phases.
    phases = np.stack([np.minimum(red, repeat), yellow, green]).mean(-1)
    threshold = np.maximum(20.0, 5.0 * noise)
    masks = []
    for index in range(3):
        other = np.max(np.delete(phases, index, axis=0), axis=0)
        masks.append(phases[index] - other >= threshold)
    return np.stack(masks)


def current_training_locations(allowed, reserved, calibration, component_validation):
    """Intersect current TRAIN geography with every later explicit reservation."""
    return {
        town: sorted(
            set(junctions)
            - set(reserved.get(town, []))
            - set(calibration.get(town, []))
            - set(component_validation.get(town, []))
        )
        for town, junctions in allowed.items()
        if town.removesuffix("_Opt") != "Town05"
    }

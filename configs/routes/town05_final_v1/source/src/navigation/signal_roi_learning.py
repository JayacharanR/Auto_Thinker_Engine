"""TRAIN-only ROI augmentation, feature-cache identity and prospective selection."""

import hashlib
import json

import numpy as np


def roi_variants(box, rng, *, positive):
    source = np.asarray(box, dtype=np.float32)
    if (
        source.shape != (4,)
        or not np.isfinite(source).all()
        or (source[2:] <= source[:2]).any()
        or (source[:2] < 0).any()
        or source[2] > 512
        or source[3] > 288
    ):
        raise ValueError("Finite positive-area native ROI required")
    result = [source.tolist()]
    w, h = source[2:] - source[:2]
    for _ in range(3):
        delta = rng.uniform(0, 0.25 if positive else 0.1, 4) * [w, h, w, h]
        candidate = source + delta * ([-1, -1, 1, 1] if positive else [1, 1, -1, -1])
        candidate[[0, 2]] = np.clip(candidate[[0, 2]], 0, 512)
        candidate[[1, 3]] = np.clip(candidate[[1, 3]], 0, 288)
        result.append(candidate.astype(np.float32).tolist())
    return result


def supervised_rois(data, index, rng):
    sample = data.base.samples[index]
    if sample["role"] != "train" or index not in data.base.train:
        raise ValueError("Only TRAIN images may enter ROI feature cache")
    rows = []
    for positive, key in [(True, "boxes"), (False, "hard_negative_boxes")]:
        for source_id, box in enumerate(data.targets[index][key]):
            for variant, roi in enumerate(roi_variants(box, rng, positive=positive)):
                rows.append(
                    dict(
                        sample_index=index,
                        label=int(positive),
                        source_id=source_id,
                        variant=variant,
                        box=roi,
                        domain=int("supplement_corpus" in sample),
                    )
                )
    return rows


def validation_payload_digest(data):
    h = hashlib.sha256()
    for i in data.base.validation:
        h.update(data.base.images[i].tobytes())
        h.update(json.dumps(data.base.samples[i], sort_keys=True).encode())
        for key, value in sorted(data.targets[i].items()):
            h.update(key.encode())
            h.update(value.tobytes())
        h.update(json.dumps(data.annotation_rows[i], sort_keys=True).encode())
    return h.hexdigest()


def operational_key(result, baseline):
    """New run only: strict improvement plus global, town and tiny-object floors."""
    current, old = result["one_to_one"], baseline["one_to_one"]
    backgrounds = result["counts"]["approximate_background_candidates"]
    if (
        current["denominator"] != old["denominator"]
        or current["iou3"] < old["iou3"]
        or current["iou3_state_correct"] < old["iou3_state_correct"]
        or backgrounds > baseline["counts"]["approximate_background_candidates"]
        or result["by_native_height"]["under8px"][0] < baseline["by_native_height"]["under8px"][0]
        or current["by_town"].keys() != old["by_town"].keys()
        or any(
            current["by_town"][town][1] != value[1] or current["by_town"][town][0] < value[0]
            for town, value in old["by_town"].items()
        )
    ):
        return None
    if (current["iou3"], current["iou3_state_correct"]) == (old["iou3"], old["iou3_state_correct"]):
        return None  # Background suppression alone is not sufficient.
    return current["iou3_state_correct"], current["iou3"], -backgrounds

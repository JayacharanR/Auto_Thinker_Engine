#!/usr/bin/env python3
"""Post-hoc CPU diagnostic: offline speed dependence, never a model selector."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_spatial_student import validate  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.spatial_data import SpatialCorpus  # noqa: E402
from src.navigation.spatial_student import SpatialCameraStudent  # noqa: E402


class Probe(torch.nn.Module):
    scratch = None

    def __init__(self, model, variant):
        super().__init__()
        self.model, self.variant = model, variant

    def forward(self, **inputs):
        if self.variant in ("zero_speed", "speed_5mps"):
            inputs["ego"] = inputs["ego"].clone()
            inputs["ego"][..., 0] = 0 if self.variant == "zero_speed" else 5
        if self.variant == "blank_rgb_features":
            inputs["features"] = torch.zeros_like(inputs["features"])
            inputs["optical"] = torch.zeros_like(inputs["optical"])
        return self.model(**inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Keep prior diagnostic")
    torch.set_num_threads(2)
    corpus = SpatialCorpus(ROOT / "outputs/navigation_spatial_data_20261019")
    state = torch.load(args.fit / "best_policy.pt", map_location="cpu", weights_only=False)
    if state["encoder_kind"] in ("cnn_scratch", "nav_color_only"):
        raise ValueError("This probe requires a frozen-encoder cache")
    model = SpatialCameraStudent(
        state["encoder_dim"], state["hidden"], color_enabled=state["color_enabled"]
    )
    model.load_state_dict(state["model"], strict=True)
    cache = json.loads((args.cache / "manifest.json").read_text())
    if digest(args.cache / "features.npy") != cache["features_sha256"]:
        raise ValueError("Feature identity changed")
    features = np.load(args.cache / "features.npy", mmap_mode="r", allow_pickle=False)
    metrics = {}
    with torch.inference_mode():
        for variant in ("observed", "zero_speed", "speed_5mps", "blank_rgb_features"):
            metrics[variant] = validate(Probe(model, variant), corpus, "cpu", features)
            print(json.dumps(dict(variant=variant, metrics=metrics[variant])), flush=True)
    strata = {}
    for role in ("train", "validation"):
        entries = [e for e in corpus.manifest["episodes"] if e["role"] == role]
        total = launch = stopped = excluded_launch = 0
        for e in entries:
            start, end = e["start"], e["end"]
            speed = corpus.arrays["ego"][start:end, 0]
            target = corpus.arrays["current_speed"][start:end]
            launch_mask = (speed < 0.2) & (target > 1)
            launch += int(launch_mask.sum())
            excluded_launch += int(launch_mask[:8].sum())
            stopped += int(((speed < 0.2) & (target < 0.1)).sum())
            total += end - start
        strata[role] = dict(
            frames=total,
            stationary_go_labels=launch,
            stationary_stop_labels=stopped,
            stationary_go_in_burnin8=excluded_launch,
        )
    atomic_json(
        args.out,
        dict(
            schema="spatial-speed-posthoc-v1",
            policy_sha256=digest(args.fit / "best_policy.pt"),
            cache_manifest_sha256=digest(args.cache / "manifest.json"),
            script_sha256=digest(__file__),
            metrics=metrics,
            strata=strata,
            note=(
                "Artificial input sensitivity, not a physical counterfactual "
                "or valid driving result. "
                "Post-hoc after pilot launch; no reselection/tuning on heldout_dev or Town05."
            ),
        ),
    )


if __name__ == "__main__":
    main()

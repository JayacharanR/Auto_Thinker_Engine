#!/usr/bin/env python3
"""CPU-only sampler/support audit; no fitting, threshold selection or simulation."""

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def original_sampler():
    old = ROOT / "outputs/navigation_m2_stop_go_20261016"
    request = json.loads((old / "request.json").read_text())
    path = old / "source/src/navigation/learning_data.py"
    if digest(path) != request["source_hashes"]["src/navigation/learning_data.py"]:
        raise ValueError("Archived original sampler changed")
    spec = importlib.util.spec_from_file_location("archived_original_navigation_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = module.ClipCache(request["cache"], request["clip_length"])
    state = torch.load(old / "latest.pt", map_location="cpu", weights_only=False)
    rng = np.random.default_rng(request["seed"])
    fractions = []
    for _ in range(state["step"]):
        starts = cache.sample(rng, request["batch_size"])
        indices = np.array(starts)[:, None] + np.arange(16, 32)[None]
        fractions.append(float((cache.arrays["current_speed"][indices] < 0.1).mean()))
    if rng.bit_generator.state != state["rng"]["sampler"]:
        raise ValueError("Original sampler replay does not reach the saved exact RNG state")
    return dict(
        updates=state["step"],
        exact_rng_state_match=True,
        stop_fraction_all=float(np.mean(fractions)),
        stop_fraction_through_selected1500=float(np.mean(fractions[:1500])),
        sampler_sha256=digest(path),
    )


def auc(truth, score):
    return float(roc_auc_score(truth, score)) if len(set(truth)) == 2 else None


def main():
    torch.set_num_threads(2)
    root = ROOT / "outputs/navigation_m2_signal_policy_cache_20261018"
    result = dict(
        kind="Post-hoc diagnosis only; no fit or validation threshold tuning",
        original_sampler=original_sampler(),
        domains={},
    )
    for domain in ("original", "approach"):
        data = root / domain
        manifest = json.loads((data / "manifest.json").read_text())
        arrays = {
            k: np.load(data / f"{k}.npy", mmap_mode="r", allow_pickle=False)
            for k in (
                "current_speed",
                "brake",
                "stop_logits",
                "ego",
                "signal_facts",
                "signal_valid",
            )
        }
        grouped = defaultdict(list)
        for e in manifest["episodes"]:
            length = e["end"] - e["start"]
            mask = np.arange(8, length)
            if e["role"] == "validation":
                mask = np.concatenate(
                    [np.arange(s + 16, s + 32) for s in range(0, length - 31, 32)]
                )
            at = e["start"] + mask
            speed, brake = arrays["current_speed"][at], arrays["brake"][at]
            stop, go = speed < 0.1, (speed > 1) & (brake < 0.5)
            features = arrays["signal_facts"][at]
            valid = arrays["signal_valid"][at]
            red = (features[..., 4] * features[..., 5] * valid).max(-1)
            green = (features[..., 4] * features[..., 7] * valid).max(-1)
            grouped[(e["role"], e["family"])].append(
                dict(
                    route_id=e["route_id"],
                    samples=len(at),
                    start_teacher_stopped=bool(arrays["current_speed"][e["start"]] < 0.1),
                    stop=int(stop.sum()),
                    go=int(go.sum()),
                    parent_stop_hits=int(((arrays["stop_logits"][at, 0] >= 0) & stop).sum()),
                    truth=stop[stop | go].tolist(),
                    red=red[stop | go].tolist(),
                    green=green[stop | go].tolist(),
                    parent_logits=arrays["stop_logits"][at[stop | go], 0].tolist(),
                    first8_red_score=float(
                        (
                            arrays["signal_facts"][e["start"] : e["start"] + 8, :, 4]
                            * arrays["signal_facts"][e["start"] : e["start"] + 8, :, 5]
                        ).max()
                    ),
                    stopped_red_mean=float(red[stop].mean()) if stop.any() else None,
                    stopped_green_mean=float(green[stop].mean()) if stop.any() else None,
                )
            )
        report = {}
        for (role, family), episodes in grouped.items():
            labels = [v for e in episodes for v in e["truth"]]
            entry = dict(
                episodes=len(episodes),
                samples=sum(e["samples"] for e in episodes),
                stop=sum(e["stop"] for e in episodes),
                go=sum(e["go"] for e in episodes),
                starts_already_teacher_stopped=sum(e["start_teacher_stopped"] for e in episodes),
                parent_stop_hits=sum(e["parent_stop_hits"] for e in episodes),
                auc={
                    k: auc(labels, [v for e in episodes for v in e[k]])
                    for k in ("red", "green", "parent_logits")
                },
                episode_details=[
                    {
                        k: v
                        for k, v in e.items()
                        if k not in ("truth", "red", "green", "parent_logits")
                    }
                    for e in episodes
                ],
            )
            report[f"{role}:{family}"] = entry
        result["domains"][domain] = report
    result.update(
        cache_manifest_sha256=digest(root / "manifest.json"),
        source_sha256=digest(__file__),
        note=(
            'Any-lamp RGB maxima are diagnostics, NOT ego-route applicability '
            'or verified visibility. AUC is not a deployed rule, threshold or '
            'driving metric.'
        ),
    )
    output = ROOT / "reports/navigation_m2_stop_generalization_20261018.json"
    atomic_json(output, result)
    print(
        json.dumps(
            dict(
                original_sampler=result["original_sampler"],
                groups={
                    d: {
                        k: {q: v for q, v in a.items() if q != "episode_details"}
                        for k, a in r.items()
                    }
                    for d, r in result["domains"].items()
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

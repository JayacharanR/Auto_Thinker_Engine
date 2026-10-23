#!/usr/bin/env python3
"""Post-hoc family errors and exact sampler-label mix; no fitting or reselection."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.causal_learning import (  # noqa: E402
    EpisodeCache,
    original_training_episodes,
    stream_forward,
)
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.student import make_student  # noqa: E402


@torch.inference_mode()
def per_family(path, cache):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = make_student(state["architecture"], state["hidden"]).cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    result = defaultdict(lambda: dict(labels=0, stop=0, stop_hit=0, go=0, false_hold=0))
    for e in cache.manifest["episodes"]:
        if e["role"] != "validation":
            continue
        x, t = cache.episode(e, "cuda")
        p = stream_forward(model, x)
        mask = torch.zeros(e["end"] - e["start"], dtype=torch.bool, device="cuda")
        for start in range(0, len(mask) - 31, 32):
            mask[start + 16 : start + 32] = True
        stop = t["current_speed"][:, mask] < 0.1
        go = (t["current_speed"][:, mask] > 1) & (t["brake"][:, mask] < 0.5)
        held = (p["current_speed"][:, mask] < 0.15) | (p["brake"][:, mask] >= 0)
        a = result[e["family"]]
        for k, v in dict(
            labels=stop.numel(),
            stop=stop.sum().item(),
            stop_hit=(held & stop).sum().item(),
            go=go.sum().item(),
            false_hold=(held & go).sum().item(),
        ).items():
            a[k] += v
    return dict(result)


def main():
    torch.set_num_threads(2)
    out = ROOT / "outputs/navigation_m2_causal_history_20261018"
    request = json.loads((out / "request.json").read_text())
    if json.loads((out / "progress.json").read_text())["status"] != "complete":
        raise ValueError("Completed fit required")
    if any(digest(out / "source" / p) != h for p, h in request["source_hashes"].items()):
        raise ValueError("Archived fit source changed")
    base = EpisodeCache(ROOT / "outputs/navigation_m2_cache_20k_20261015")
    extra = EpisodeCache(Path(request["cache"]))
    guard = json.loads((Path(request["collection"]) / "guard.json").read_text())
    pool, _ = original_training_episodes(base.manifest, guard["allowed_junctions"])
    families = sorted({e["family"] for e in pool})
    rng = np.random.default_rng(42)
    trace = [json.loads(s) for s in (out / "train.jsonl").read_text().splitlines()]
    mix = defaultdict(list)
    for step, logged in enumerate(trace):
        if step % 2:
            entries = extra.manifest["episodes"]
            e = entries[int(rng.integers(len(entries)))]
            arrays = extra.arrays
            domain = "approach"
        else:
            family = families[int(rng.integers(len(families)))]
            entries = [e for e in pool if e["family"] == family]
            e = entries[int(rng.integers(len(entries)))]
            arrays = base.arrays
            domain = "original"
        if (
            e["route_id"] != logged["route_id"]
            or e["end"] - e["start"] != logged["frames"]
            or logged["domain"] != domain
        ):
            raise ValueError("Sampler replay differs from actual training log")
        labels = arrays["current_speed"][e["start"] + 8 : e["end"]]
        mix[domain].append(float(np.mean(labels < 0.1)))
    check_resources(ROOT / "outputs/navigation_m2_causal_diagnosis_20261018", 25)
    parent = Path(request["parent"])
    latest = out / "latest.pt"
    stats = dict(parent=per_family(parent, base), rejected_latest1000=per_family(latest, base))
    result = dict(
        kind="Post-hoc diagnostics; no fit or new selection",
        family=stats,
        exact_sampler_updates_replayed=len(trace),
        stop_label_fraction_equal_update_weight={k: float(np.mean(v)) for k, v in mix.items()},
        combined_stop_label_fraction=float(np.mean([x for v in mix.values() for x in v])),
        source_sha256=digest(Path(__file__)),
        parent_sha256=digest(parent),
        latest_sha256=digest(latest),
        request_sha256=digest(out / "request.json"),
        visibility_labels_approved=False,
    )
    atomic_json(ROOT / "reports/navigation_m2_causal_diagnosis_20261018.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation.lock"):
        main()

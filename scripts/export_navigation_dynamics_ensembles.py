#!/usr/bin/env python3
"""Export completed factual members without asserting planning/risk qualification."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.dynamics_inference import INFERENCE_SOURCES  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--encoder-descriptor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Retain every factual-model export generation")
    if read(args.study / "progress.json")["status"] != "complete":
        raise ValueError("Finish every registered factual member before export")
    feature_manifest = args.features / "manifest.json"
    feature_request = read(args.features / "request.json")
    if (digest(args.encoder_descriptor) != feature_request["descriptor_sha256"]
            or read(args.encoder_descriptor) != feature_request["descriptor"]):
        raise ValueError("Export must retain the actual training encoder identity")
    descriptors = []
    for model in ("jepa", "rssm"):
        members = []
        for seed in (42, 1051, 2060):
            root = args.study / f"{model}_seed{seed}"
            progress = read(root / "progress.json")
            request_path = root / "request.json"
            request = read(request_path)
            if (
                progress["status"] != "complete" or progress["steps"] != 2000
                or request["seed"] != seed or request["model"] != model
                or request["encoder_cache_manifest_sha256"] != digest(feature_manifest)
                or request["red_training_masked"] is not True
            ):
                raise ValueError("All three complete matched factual members are required")
            log = [json.loads(row) for row in (root / "train.jsonl").read_text().splitlines()]
            validations = [json.loads(row) for row in
                           (root / "validation.jsonl").read_text().splitlines()]
            if ([r["step"] for r in log] != list(range(1, 2001))
                    or [r["step"] for r in validations] != [0, 500, 1000, 1500, 2000]):
                raise ValueError("Actual applied update/validation evidence is incomplete")
            checkpoint = root / "best_model.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (state["request_sha256"] != digest(request_path)
                    or state["step"] != progress["selected_step"]
                    or state["red_risk_qualified"] is not False
                    or state["planner_qualified"] is not False):
                raise ValueError("Selected member does not match its completed registered fit")
            members.append(dict(seed=seed, checkpoint=str(checkpoint.resolve()),
                                sha256=digest(checkpoint), selected_step=state["step"],
                                request_sha256=digest(request_path),
                                train_log_sha256=digest(root / "train.jsonl"),
                                validation_sha256=digest(root / "validation.jsonl")))
        descriptors.append(dict(
            schema="navigation-factual-ensemble-v1", model=model, members=members,
            encoder_descriptor=str(args.encoder_descriptor.resolve()),
            encoder_descriptor_sha256=digest(args.encoder_descriptor),
            feature_manifest_sha256=digest(feature_manifest),
            study_request_sha256=digest(args.study / "request.json"),
            red_risk_qualified=False, planner_qualified=False, calibration_qualified=False,
            independent_members=True, inference_source_sha256={p: digest(ROOT / p)
                                                              for p in INFERENCE_SOURCES},
            no_new_updates=True,
        ))
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / "export_source.py")
    for descriptor in descriptors:
        atomic_json(args.out / (descriptor["model"] + ".json"), descriptor)
    atomic_json(args.out / "assessment.json", dict(
        models=2, members_per_model=3, all_training_evidence_verified=True,
        zero_positive_red_limitation_retained=True, planner_qualified=False,
        source_sha256=digest(__file__),
    ))
    print(json.dumps(dict(ensembles=2, members=6, planner_qualified=False)), flush=True)


if __name__ == "__main__":
    main()

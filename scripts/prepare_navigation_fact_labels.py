#!/usr/bin/env python3
"""Separate full-label M2 fact targets, never inherited by M4 label-budget arms."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.spatial_data import DOMAINS  # noqa:E402

FACTS = ("light", "presence", "distance", "hazards")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve every fact-label preparation")
    data = ROOT / "outputs/navigation_spatial_data_repaired_20261019"
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["samples"] != 18534 or manifest["annotation_budget"]["total_labels"] != 15712:
        raise ValueError("Only the registered corrected full-label corpus is allowed")
    arrays = {name: {} for name in DOMAINS}
    sources = [
        Path(__file__),
        ROOT / "src/navigation/fact_student.py",
        ROOT / "reports/navigation_fact_student_protocol_20261020.md",
    ]
    for domain, root in DOMAINS.items():
        if digest(root / "manifest.json") != manifest["source_manifest_sha256"][domain]:
            raise ValueError("Preserved raw-label manifest changed")
        for name in FACTS:
            arrays[domain][name] = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    required = sum(
        manifest["samples"] * a.dtype.itemsize * int(np.prod(a.shape[1:]))
        for a in arrays["original"].values()
    )
    if shutil.disk_usage(args.out.parent).free < 25e9 + required + 1e6:
        raise RuntimeError("Fact arrays would breach25GB reserve")
    args.out.mkdir(parents=True)
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    request = dict(
        corrected_data_manifest_sha256=digest(data / "manifest.json"),
        full_training_labels=15712,
        validation_labels=2822,
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        source_fact_array_sha256={
            str(DOMAINS[d] / f"{name}.npy"): digest(DOMAINS[d] / f"{name}.npy")
            for d in DOMAINS
            for name in FACTS
        },
        raw_label_arrays_only=True,
        no_training_updates=True,
        not_part_of_m4_label_efficiency=True,
        visibility_certified=False,
    )
    atomic_json(args.out / "request.json", request)
    outputs = {}
    for name in FACTS:
        example = arrays["original"][name]
        result = np.empty((manifest["samples"], *example.shape[1:]), dtype=example.dtype)
        for e in manifest["episodes"]:
            result[e["start"] : e["end"]] = arrays[e["domain"]][name][
                e["source_start"] : e["source_end"]
            ]
        np.save(args.out / f"{name}.npy", result, allow_pickle=False)
        outputs[name] = result
    train_ids = np.concatenate(
        [np.arange(e["start"], e["end"]) for e in manifest["episodes"] if e["role"] == "train"]
    )
    summary = dict(
        request_sha256=digest(args.out / "request.json"),
        arrays_sha256={f"{name}.npy": digest(args.out / f"{name}.npy") for name in FACTS},
        full_training_labels=len(train_ids),
        light_class_counts_train=np.bincount(outputs["light"][train_ids], minlength=5).tolist(),
        presence_counts_train=outputs["presence"][train_ids].sum(0).tolist(),
        hazard_counts_train=outputs["hazards"][train_ids].sum(0).tolist(),
        original_validation_unchanged=True,
        driving_claim=False,
    )
    atomic_json(args.out / "manifest.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

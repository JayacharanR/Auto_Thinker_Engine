#!/usr/bin/env python3
"""Additional random-ViT diagnostic, using the completed study's unchanged train loop."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_visual_student import train  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("One fixed diagnostic attempt; no blind repeat")
    main_request = json.loads(
        (ROOT / "outputs/navigation_visual_study_20261019/request.json").read_text()
    )
    source = "scripts/train_navigation_visual_student.py"
    if digest(ROOT / source) != main_request["source_hashes"][source]:
        raise ValueError("Diagnostic must use the exact completed main training loop")
    args.out.mkdir(parents=True)
    for p in [Path(__file__), ROOT / "reports/navigation_random_vit_protocol_20261019.md"]:
        target = args.out / "diagnostic_source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    atomic_json(
        args.out / "diagnostic_request.json",
        dict(
            wrapper_sha256=digest(__file__),
            protocol_sha256=digest(ROOT / "reports/navigation_random_vit_protocol_20261019.md"),
            main_study_request_sha256=digest(
                ROOT / "outputs/navigation_visual_study_20261019/request.json"
            ),
            exact_training_loop_sha256=digest(ROOT / source),
            random_backbone_frozen=True,
            local_pretraining_labels=0,
            pretraining_updates=0,
        ),
    )
    train(
        SimpleNamespace(
            encoder="jepa_random",
            cache=args.cache,
            out=args.out,
            data=ROOT / "outputs/navigation_spatial_data_20261019",
            fraction=1.0,
            seed=42,
            steps=2000,
            max_wall=3600.0,
            no_color=False,
            resume=False,
        )
    )


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation.lock"):
        main()

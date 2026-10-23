#!/usr/bin/env python3
"""Fetch official baseline weights into the workspace and bind their SHA hashes."""

import json
import socket
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.grid_encoders import file_sha256  # noqa: E402
from src.navigation.runtime import atomic_json, check_resources  # noqa: E402


def main():
    root = ROOT / "outputs/navigation_models"
    check_resources(root, 25)
    urls = {
        "imagenet_resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "dinov2_small": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    }
    manifest = root / "navigation_backbones_20261019.json"
    known = json.loads(manifest.read_text()) if manifest.exists() else {}
    socket.setdefaulttimeout(60)
    for kind, url in urls.items():
        name = url.rsplit("/", 1)[-1]
        path = root / name
        if kind in known:
            if not path.exists() or file_sha256(path) != known[kind]["sha256"]:
                raise ValueError("Previously registered official weight changed")
        else:
            torch.hub.load_state_dict_from_url(
                url,
                model_dir=str(root),
                check_hash=kind == "imagenet_resnet18",
                progress=False,
                weights_only=True,
            )
            known[kind] = dict(
                kind=kind,
                checkpoint=str(path),
                sha256=file_sha256(path),
                official_url=url,
                local_carla_labels=0,
                external_pretraining="ImageNet1K supervised"
                if kind == "imagenet_resnet18"
                else "DINOv2 LVD142M self-supervised",
            )
            atomic_json(manifest, known)
        print(json.dumps(known[kind]), flush=True)
    path = ROOT / "outputs/checkpoints/phase2_sigreg/final.pt"
    known["jepa_comma2k19"] = dict(
        kind="jepa_comma2k19",
        checkpoint=str(path),
        sha256=file_sha256(path),
        local_carla_labels=0,
        external_pretraining="Project comma2k19 single-chunk JEPA SigReg, ViT-S/16,128px,8frames",
    )
    atomic_json(manifest, known)


if __name__ == "__main__":
    main()

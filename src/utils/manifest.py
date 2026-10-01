"""
Run manifest utility for auditable experiments (P0 Contract).

Captures git revisions, config hashes, hardware/environment metadata,
seed, CARLA details, and checkpoint provenance to ensure full reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _get_git_commit(repo_path: Path) -> str:
    """Safely get git commit hash for a repository or submodule."""
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            diff_res = subprocess.run(
                ["git", "-C", str(repo_path), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if diff_res.returncode == 0 and diff_res.stdout.strip():
                commit += "-dirty"
            return commit
    except Exception:
        pass
    return "unknown"


def _hash_config(config: Any) -> str:
    """Compute SHA256 hash of a config dictionary or string."""
    try:
        if isinstance(config, (dict, list)):
            data = json.dumps(config, sort_keys=True).encode("utf-8")
        elif isinstance(config, str):
            data = config.encode("utf-8")
        else:
            data = str(config).encode("utf-8")
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return "unknown"


def create_run_manifest(
    config: Optional[dict] = None,
    arm: str = "cnn",
    seed: int = 42,
    task: str = "carla_right_turn_simple",
    checkpoint_path: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    output_path: Optional[Path | str] = None,
) -> dict:
    """
    Create a comprehensive audit manifest for a training or evaluation run.

    Args:
        config: Full configuration dictionary for the run.
        arm: Model arm identifier (e.g. 'cnn', 'custom_jepa', 'vjepa2').
        seed: Random seed used.
        task: CARLA environment task name.
        checkpoint_path: Path to checkpoint if resuming or evaluating.
        extra_metadata: Additional arbitrary metadata.
        output_path: If provided, write manifest to this JSON file.

    Returns:
        Manifest dictionary.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    submodule_cardreamer = project_root / "third_party" / "CarDreamer"
    submodule_dreamerv3 = project_root / "third_party" / "dreamerv3_torch"

    gpu_info = {}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                "cuda_available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "total_memory_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2
                ),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
        else:
            gpu_info = {"cuda_available": False, "torch_version": torch.__version__}
    except Exception as e:
        gpu_info = {"error": str(e)}

    manifest = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": arm,
        "seed": seed,
        "task": task,
        "config_hash": _hash_config(config) if config else None,
        "checkpoint": {
            "path": str(checkpoint_path) if checkpoint_path else None,
            "exists": os.path.exists(checkpoint_path) if checkpoint_path else False,
        },
        "git": {
            "root_commit": _get_git_commit(project_root),
            "cardreamer_commit": _get_git_commit(submodule_cardreamer),
            "dreamerv3_torch_commit": _get_git_commit(submodule_dreamerv3),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "carla_root": os.environ.get("CARLA_ROOT", "not set"),
            "carla_port": int(os.environ.get("CARLA_PORT", 2000)),
            "carla_version": "0.9.15",
            "gpu": gpu_info,
        },
        "extra": extra_metadata or {},
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest

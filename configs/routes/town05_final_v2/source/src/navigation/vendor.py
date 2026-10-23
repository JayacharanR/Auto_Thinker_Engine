"""Load only the pinned evaluation/expert checkout, with a tracked patch."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def activate(carla_root=None):
    lock = json.loads((PROJECT_ROOT / "configs/navigation_vendor.json").read_text())
    root = PROJECT_ROOT / lock["directory"]
    if not (root / ".git").exists():
        raise RuntimeError("Run scripts/setup_navigation.py first")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != lock["revision"]:
        raise RuntimeError(f"Unexpected vendor revision: {revision}")
    patch = PROJECT_ROOT / lock["patch"]
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "--"])
    if diff != patch.read_bytes():
        raise RuntimeError("Vendor diff differs from the tracked compatibility patch")
    carla_root = Path(carla_root or os.environ.get("CARLA_ROOT", "~/software/CARLA_0.9.15"))
    carla_root = carla_root.expanduser().resolve()
    paths = [
        PROJECT_ROOT / lock["python_target"],
        root / "team_code",
        root / "leaderboard_autopilot",
        root / "scenario_runner_autopilot",
        carla_root / "PythonAPI/carla",
    ]
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    os.environ["CARLA_ROOT"] = str(carla_root)
    os.environ["SCENARIO_RUNNER_ROOT"] = str(root / "scenario_runner_autopilot")
    os.environ["LEADERBOARD_ROOT"] = str(root / "leaderboard_autopilot")
    return {
        "revision": revision,
        "patch_sha256": hashlib.sha256(diff).hexdigest(),
        "root": str(root),
        "carla_root": str(carla_root),
    }

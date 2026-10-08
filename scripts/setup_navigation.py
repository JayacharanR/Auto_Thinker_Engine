#!/usr/bin/env python3
"""Install pinned M0 code/dependencies into outputs, without changing .venv."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install", action="store_true", help="Download source and missing packages"
    )
    args = parser.parse_args()
    lock = json.loads((ROOT / "configs/navigation_vendor.json").read_text())
    root = ROOT / lock["directory"]
    if args.install:
        if not (root / ".git").exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--sparse",
                    "--no-checkout",
                    lock["repository"],
                    str(root),
                ],
                check=True,
            )
            subprocess.run(["git", "-C", str(root), "checkout", lock["revision"]], check=True)
            subprocess.run(
                ["git", "-C", str(root), "sparse-checkout", "set", *lock["sparse_paths"]],
                check=True,
            )
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != lock["revision"]:
            raise RuntimeError("Existing vendor has another revision; refusing to overwrite")
        patch = ROOT / lock["patch"]
        if not subprocess.check_output(["git", "-C", str(root), "diff", "--"]):
            subprocess.run(["git", "-C", str(root), "apply", str(patch)], check=True)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(ROOT / lock["python_target"]),
                "--no-deps",
                *lock["dependencies"],
            ],
            check=True,
        )
    from src.navigation.vendor import activate

    print(json.dumps(activate(), indent=2))
    # Import the actual entrypoints, not just their package names.
    from autopilot import AutoPilot
    from leaderboard.leaderboard_evaluator_local import LeaderboardEvaluator
    from srunner.scenariomanager.scenarioatomics.atomic_criteria import RunningStopTest

    print(
        "Imports OK:", AutoPilot.__name__, LeaderboardEvaluator.__name__, RunningStopTest.__name__
    )


if __name__ == "__main__":
    main()

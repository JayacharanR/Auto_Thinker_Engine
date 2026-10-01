#!/usr/bin/env python3
"""Single terminal entry point for the driving world-model project.

This file is intentionally a launcher, not a copy of every model class. It
keeps the existing phase scripts and configuration files as the source of
truth while providing one stable command for a remote or headless server.

Examples:
    python run.py doctor
    python run.py smoke --host localhost --port 2000
    python run.py phase2 --config configs/phase2_jepa_pretrain.yaml
    python run.py train --arm cnn --steps 10000
    python run.py compare --steps 500000
    python run.py probe --checkpoint outputs/checkpoints/phase2/best.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DREAMER_DIR = PROJECT_ROOT / "third_party" / "dreamerv3_torch"
CARDREAMER_DIR = PROJECT_ROOT / "third_party" / "CarDreamer"

FORWARDING_SCRIPTS = {
    "smoke": "smoke_test_carla.py",
    "phase2": "train_phase2_jepa.py",
    "train": "train_cardreamer.py",
    "evaluate": "evaluate_agent.py",
    "probe": "probe_phase2.py",
    "visualize": "visualize_representations.py",
    "download": "download_comma2k19.py",
}


def _runtime_environment() -> dict[str, str]:
    """Return a subprocess environment with local project paths configured."""
    environment = os.environ.copy()
    paths = [str(PROJECT_ROOT), str(DREAMER_DIR), str(CARDREAMER_DIR)]
    carla_root = environment.get("CARLA_ROOT")
    if carla_root:
        paths.insert(0, str(Path(carla_root) / "PythonAPI" / "carla"))

    old_pythonpath = environment.get("PYTHONPATH")
    if old_pythonpath:
        paths.append(old_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


def _add_forwarding_command(subparsers, name: str, description: str):
    """Add a command whose remaining arguments are passed to a phase script."""
    command = subparsers.add_parser(name, description=description)
    command.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the underlying project script.",
    )
    return command


def _run_script(script_name: str, script_args: list[str]) -> int:
    """Run one of the existing scripts from the repository root."""
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.is_file():
        print(f"[run] ERROR: script not found: {script_path}", file=sys.stderr)
        return 2

    command = [sys.executable, str(script_path), *script_args]
    print(f"[run] Executing: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_runtime_environment(),
    )
    return completed.returncode


def _check_import(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _doctor(args: argparse.Namespace) -> int:
    """Check the local project and optional server runtime prerequisites."""
    print(f"[doctor] Project: {PROJECT_ROOT}")
    failures = 0

    def report(label: str, ok: bool, detail: str, required: bool = False) -> None:
        nonlocal failures
        state = "OK" if ok else ("MISSING" if required else "WARN")
        print(f"  {state:7} {label}: {detail}")
        if required and not ok:
            failures += 1

    running_python_is_supported = sys.version_info >= (3, 10) and sys.version_info < (3, 11)
    report("Python", running_python_is_supported, sys.version.split()[0], True)
    report("dreamerv3-torch", (DREAMER_DIR / "models.py").is_file(), str(DREAMER_DIR), True)
    report(
        "CarDreamer",
        (CARDREAMER_DIR / "car_dreamer").is_dir(),
        str(CARDREAMER_DIR),
        args.require_carla,
    )

    required_modules = ["torch", "numpy", "yaml", "ruamel.yaml"]
    if args.require_carla:
        required_modules.extend(["flask", "cv2", "shapely"])

    for module_name in required_modules:
        importable = _check_import(module_name)
        report(
            f"Python module {module_name}",
            importable,
            "importable" if importable else "not importable",
            True,
        )

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_detail = (
            f"available ({torch.cuda.device_count()} device(s))"
            if cuda_available
            else "unavailable"
        )
    except Exception as error:  # pragma: no cover - depends on installed runtime
        cuda_available = False
        cuda_detail = f"torch import failed: {error}"
    report("CUDA", cuda_available, cuda_detail, args.require_cuda)

    carla_root = os.environ.get("CARLA_ROOT")
    carla_server = bool(carla_root and (Path(carla_root) / "CarlaUE4.sh").is_file())
    carla_module = _check_import("carla")
    report("CARLA_ROOT", carla_server, carla_root or "not set", args.require_carla)
    report(
        "CARLA Python API",
        carla_module,
        "importable" if carla_module else "not importable",
        args.require_carla,
    )

    if args.data_dir:
        data_path = Path(args.data_dir).expanduser()
        report("Dataset", data_path.is_dir(), str(data_path), args.require_data)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser()
        report("Checkpoint", checkpoint_path.is_file(), str(checkpoint_path), True)

    if failures:
        print(f"[doctor] FAILED: {failures} required check(s) did not pass.")
        return 1

    print("[doctor] Passed. Warnings above are only blockers for commands that require them.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One terminal entry point for the driving world-model project."
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="Check runtime prerequisites.")
    doctor.add_argument("--require-cuda", action="store_true")
    doctor.add_argument("--require-carla", action="store_true")
    doctor.add_argument("--require-data", action="store_true")
    doctor.add_argument("--data-dir", default=None)
    doctor.add_argument("--checkpoint", default=None)

    _add_forwarding_command(subparsers, "smoke", "Run the CARLA connectivity smoke test.")
    _add_forwarding_command(subparsers, "phase2", "Run JEPA pretraining.")
    _add_forwarding_command(
        subparsers,
        "train",
        "Train one Phase 1/3 arm with the active Dreamer runner.",
    )
    _add_forwarding_command(
        subparsers,
        "evaluate",
        "Evaluate a trained driving agent in CARLA with video and metrics.",
    )
    _add_forwarding_command(subparsers, "compare", "Run the Phase 3 three-arm comparison.")
    _add_forwarding_command(subparsers, "probe", "Run the Phase 2 linear probe.")
    _add_forwarding_command(subparsers, "visualize", "Visualize Phase 2 representations.")
    _add_forwarding_command(subparsers, "download", "Download or verify comma2k19 data.")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    # Forward the complete argument tail without argparse trying to interpret
    # phase-specific flags such as --arm, --steps, or --checkpoint.
    if raw_args and raw_args[0] in FORWARDING_SCRIPTS:
        return _run_script(FORWARDING_SCRIPTS[raw_args[0]], raw_args[1:])
    if raw_args and raw_args[0] == "compare":
        return _run_script("train_cardreamer.py", ["--comparison", *raw_args[1:]])

    args = parser.parse_args(raw_args)

    if args.command is None:
        parser.print_help()
        return 2

    if args.command == "doctor":
        return _doctor(args)

    # Forwarding commands return above. This branch is retained as a guard in
    # case another forwarding command is added without updating main().
    if args.command in FORWARDING_SCRIPTS:
        return _run_script(FORWARDING_SCRIPTS[args.command], args.script_args)
    if args.command == "compare":
        return _run_script("train_cardreamer.py", ["--comparison", *args.script_args])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

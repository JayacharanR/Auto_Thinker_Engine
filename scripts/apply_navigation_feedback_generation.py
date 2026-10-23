#!/usr/bin/env python3
"""Apply approved own-actuator interface after pinned work, then audit one TRAIN probe."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.apply_navigation_compact_generation import live_dependencies  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402

DEPS = [
    ROOT / "outputs" / name
    for name in (
        "navigation_rule_followthrough_v3_20261020",
        "navigation_semantic_residual_study_v2_20261020",
        "navigation_fixed_branches_20261020",
        "navigation_matched_native_completion_20261020",
    )
]
APPROVAL = ROOT / "reports/navigation_ego_feedback_approval_20261020.json"
PARENT = ROOT / "outputs/navigation_fact_study_20261020/cnn_scratch_facts/best_policy.pt"
PARENT_SHA = "e3a688576d007158ee9d61527de83c4eed839b2bd6ddb0d1bc34e5f88181ea48"
PROBE_ROUTE = ROOT / "configs/encounters/location_split_v1/train/Town01_Opt.xml"


def read(path):
    return json.loads(Path(path).read_text())


def verify_generation(generation, root=ROOT):
    for name, identities in generation["targets"].items():
        path = root / name
        before = identities["before_sha256"]
        if before is None:
            if path.exists():
                raise ValueError(f"New generation target already exists:{name}")
        elif not path.is_file() or digest(path) != before:
            raise ValueError(f"Pinned generation target changed:{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.generation = args.generation.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    generation = read(args.generation / "generation.json")
    validation = read(args.generation / "validation.json")
    approval = read(APPROVAL)
    if (
        not approval["user_approved"]
        or generation["approval_sha256"] != digest(APPROVAL)
        or validation["status"] != "passed"
        or validation["generation_sha256"] != digest(args.generation / "generation.json")
        or validation["patch_sha256"] != digest(args.generation / "feedback.patch")
    ):
        raise ValueError("Only the exact approved, tested source generation can be activated")
    pinned = read(DEPS[-1] / "request.json")["inference_source_sha256"]
    frozen = dict(
        generation=digest(args.generation / "generation.json"),
        validation=digest(args.generation / "validation.json"),
        patch=digest(args.generation / "feedback.patch"),
        approval=digest(APPROVAL),
        launcher=digest(__file__),
        parent=PARENT_SHA,
        probe_route=digest(PROBE_ROUTE),
    )
    atomic_json(
        args.out / "request.json",
        dict(
            hashes=frozen,
            after=[str(p) for p in DEPS],
            inference_source_sha256=pinned,
            input_extension_only=True,
            planner_active=False,
            no_candidate_promotion=True,
            native_probe=dict(
                role="train_instrumentation_only",
                route=str(PROBE_ROUTE),
                route_id=6,
                seed=62041,
                frames=160,
                maximum_wall_seconds=240,
            ),
            no_training_updates=True,
            preserve_all_outcomes=True,
        ),
    )
    shutil.copyfile(__file__, args.out / "activation_source.py")
    status, active = "running", None

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="Approved ego-only feedback interface",
                phase=phase,
                active_output=str(active) if active else None,
                updated_at=time.time(),
            ),
        )

    def run(command, logfile, environment=None):
        if (args.out / "STOP").exists():
            raise InterruptedError("Feedback validation STOP before starting another child")
        with logfile.open("x") as log:
            child = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            while child.poll() is None:
                if (args.out / "STOP").exists() and active is not None and active.exists():
                    (active / "STOP").touch()
                progress("Validating exact source generation; no policy promotion")
                time.sleep(2)
        if (args.out / "STOP").exists():
            raise InterruptedError("Feedback validation STOP; preserve the completed child")
        if child.returncode:
            raise RuntimeError(f"Validation failed; inspect retained {logfile}")

    try:
        with BudgetWatch(args.out):
            while True:
                if (args.out / "STOP").exists():
                    raise InterruptedError("Feedback activation STOP")
                states = [read(p / "progress.json")["status"] for p in DEPS]
                if any(s not in ("running", "starting", "complete") for s in states):
                    raise RuntimeError(
                        "Pinned predecessor failed/stopped; inspect before activation"
                    )
                if all(s == "complete" for s in states) and not live_dependencies(DEPS):
                    break
                progress("Waiting for all source-pinned evaluations and workers to terminate")
                time.sleep(3)
            with RunLock(ROOT / "outputs/.navigation.lock"):
                if (args.out / "STOP").exists():
                    raise InterruptedError("Feedback activation STOP before source mutation")
                if live_dependencies(DEPS):
                    raise RuntimeError("Pinned workers still live after generation lock")
                if (
                    digest(args.generation / "generation.json") != frozen["generation"]
                    or digest(args.generation / "validation.json") != frozen["validation"]
                    or digest(args.generation / "feedback.patch") != frozen["patch"]
                    or digest(APPROVAL) != frozen["approval"]
                    or digest(PROBE_ROUTE) != frozen["probe_route"]
                ):
                    raise ValueError("Approved queued generation, validation or route changed")
                if any(digest(ROOT / p) != h for p, h in pinned.items()):
                    raise ValueError("Source-pinned predecessor generation changed")
                verify_generation(generation)
                for name in generation["targets"]:
                    path = ROOT / name
                    if path.is_file():
                        old = args.out / "before" / name
                        old.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, old)
                patch = str(args.generation / "feedback.patch")
                subprocess.run(["git", "apply", "--check", patch], cwd=ROOT, check=True)
                progress("Applying the approved versioned interface after pinned evaluations")
                subprocess.run(["git", "apply", patch], cwd=ROOT, check=True)
                actual = {p: digest(ROOT / p) for p in generation["targets"]}
                atomic_json(
                    args.out / "applied.json",
                    dict(
                        actual_after=actual,
                        targets=generation["targets"],
                        applied_at=time.time(),
                        all_pinned_owners_terminated=True,
                        no_evaluation_outcomes_changed=True,
                    ),
                )
                if any(actual[p] != h["after_sha256"] for p, h in generation["targets"].items()):
                    raise ValueError("Applied bytes differ from the validated generation")
                env = dict(
                    os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"
                )
                run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_navigation_ego_feedback.py",
                        "tests/test_navigation_student.py",
                        "tests/test_navigation_final_evaluation.py",
                        "tests/test_navigation_final_registration.py",
                    ],
                    args.out / "validation.log",
                    env,
                )
                import torch

                if digest(PARENT) != PARENT_SHA:
                    raise ValueError("Frozen parity reference parent changed")
                state = torch.load(PARENT, map_location="cpu", weights_only=True)
                state.update(
                    policy_input_schema="navigation-ego-action-step-v2",
                    feedback_parity_reference=True,
                    parent_sha256=PARENT_SHA,
                    native_deployable=False,
                )
                reference = args.out / "feedback_parity_reference.pt"
                torch.save(state, reference)
                del state
                active = args.out / "native_feedback_probe"
                run(
                    [
                        sys.executable,
                        "-u",
                        "scripts/run_navigation.py",
                        "student",
                        "--route",
                        str(PROBE_ROUTE),
                        "--routes-subset",
                        "6",
                        "--seed",
                        "62041",
                        "--checkpoint",
                        str(reference),
                        "--max-steps",
                        "160",
                        "--max-wall",
                        "240",
                        "--out",
                        str(active),
                    ],
                    args.out / "native_probe.log",
                )
                if (args.out / "STOP").exists():
                    raise InterruptedError("Feedback activation stopped after native probe")
                episodes = list((active / "episodes").iterdir())
                recordings = list((active / "recordings").glob("*.log"))
                if len(episodes) != 1 or len(recordings) != 1:
                    raise ValueError("One original instrumentation episode/recorder required")
                run(
                    [
                        sys.executable,
                        "scripts/audit_navigation_ego_feedback.py",
                        "--episode",
                        str(episodes[0]),
                        "--recorder",
                        str(recordings[0]),
                        "--out",
                        str(args.out / "native_feedback_audit"),
                    ],
                    args.out / "native_audit.log",
                )
                measured = read(args.out / "native_feedback_audit/assessment.json")
                if (
                    measured.get("observations", 0) < 2
                    or not measured["all_feedback_has_native_ack"]
                ):
                    raise ValueError("Real native actuator acknowledgments incomplete")
                from src.navigation.recorder_collisions import recorder_contacts

                contacts = recorder_contacts(recordings[0])
                atomic_json(
                    args.out / "assessment.json",
                    dict(
                        interface_applied=True,
                        unit_and_real_worker_parity_passed=True,
                        native_feedback_audit=measured,
                        unfiltered_contact_starts=contacts["ego_contact_starts"],
                        raw_contact_audit=contacts,
                        native_summary=read(active / "summary.json"),
                        planner_active=False,
                        no_model_or_safety_gate_promoted=True,
                    ),
                )
                status = "complete"
                progress(
                    "Interface/native feedback verified; trajectory planners still unqualified"
                )
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(
            args.out / "error.json",
            dict(
                error=str(error),
                at=time.time(),
                applied=(args.out / "applied.json").exists(),
                all_artifacts_retained=True,
            ),
        )
        progress(str(error))
        raise


if __name__ == "__main__":
    main()

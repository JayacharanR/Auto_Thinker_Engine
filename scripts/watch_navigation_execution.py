#!/usr/bin/env python3
"""Read-only live milestone status and current run; follows stage changes."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.watch_navigation import render  # noqa: E402


def snapshot(history=False):
    ledger = json.loads((ROOT / "reports/navigation_execution_20261018.json").read_text())
    lines = ["Navigation M0–M6 execution; current run progress is not a driving-success score."]
    paused = ledger.get("execution_status") == "paused_by_user"
    if paused:
        lines.extend(
            [
                "PAUSED BY USER: no experiment may resume without a new user request.",
                "The completed/queued records below are retained history; M6 is not achieved.",
            ]
        )
    lines.extend(f"{stage}: {data['status']}" for stage, data in ledger["milestones"].items())
    output = ledger.get("active_output")
    if output:
        lines.extend(["", output, render(ROOT / output)[0]])
        current, seen = ROOT / output, {(ROOT / output).resolve()}
        for _ in range(4):
            progress_path = current / "progress.json"
            if not progress_path.exists():
                break
            active = json.loads(progress_path.read_text()).get("active_output")
            if not active:
                break
            current = Path(active)
            if not current.is_absolute():
                current = ROOT / current
            if current.resolve() in seen:
                break
            if not (current / "progress.json").exists():
                lines.append("Current evidence directory: " + str(current))
                break
            seen.add(current.resolve())
            lines.extend(["", str(current), render(current)[0]])
        for path in sorted((ROOT / output).glob("*/chunk_*/progress.json")):
            try:
                progress = json.loads(path.read_text())
                if progress["status"] not in ("starting", "running"):
                    continue
                route_file = path.parent / "current_route.json"
                route = (
                    json.loads(route_file.read_text()).get("route_id")
                    if route_file.exists()
                    else "loading"
                )
                lines.append(
                    f"  {path.parent.name}: route {route}, "
                    f"{progress.get('route_progress_percent', 0):.1f}% | "
                    f"heartbeat {time.time() - progress.get('updated_at', 0):.0f}s ago"
                )
            except (OSError, ValueError, KeyError):
                continue  # Read-only snapshots can race an evaluator write.
    closed = {}
    displayed = {output}

    def related(path, label):
        if path in displayed:
            return
        displayed.add(path)
        progress = ROOT / path / "progress.json"
        if progress.exists():
            state = json.loads(progress.read_text()).get("status", "unknown")
            if state not in ("starting", "running"):
                closed[state] = closed.get(state, 0) + 1
                if not history:
                    return
        lines.extend(["", label, render(ROOT / path)[0]])

    resource = ledger.get("resource_output")
    if resource:
        related(resource, resource)
    queued = ledger.get("queued_output")
    if queued and queued != output:
        related(queued, "Queued: " + queued)
    later = ledger.get("queued_after_reference")
    if later:
        related(later, "After physical references: " + later)
    repaired = ledger.get("queued_repaired_study")
    if repaired:
        related(repaired, "Corrected study queued: " + repaired)
    dynamics = ledger.get("queued_dynamics_features")
    if dynamics:
        related(dynamics, "Factual encoding queued: " + dynamics)
    for key, label in (
        ("queued_vehicle_gpu_profile", "GPU perception profile queued"),
        ("queued_memory_reference", "Memory setup reference queued"),
        ("queued_dynamics_study", "Factual world-model study queued"),
        ("queued_camera_pilots", "Corrected camera pilots queued"),
        ("queued_branch_clock_pilot", "Replay clock repair queued"),
        ("queued_storage_activation", "Approved storage generation queued"),
        ("queued_rule_student", "Frozen-primary rule study queued"),
        ("queued_semantic_study", "Protected semantic countermeasure queued"),
        ("queued_fixed_branches", "Frozen-action branch pilot queued"),
        ("queued_matched_native", "Remaining matched encoder pilots queued"),
        ("queued_ego_feedback_activation", "Approved ego feedback activation queued"),
        ("queued_train_integration", "TRAIN-only long integration queued"),
        ("queued_dense_labels", "Five-town dense labels"),
        ("queued_dense_label_audit", "Independent dense-label audit"),
    ):
        path = ledger.get(key)
        if path:
            related(path, label + ": " + path)
    for auxiliary in ledger.get("auxiliary_outputs", []):
        related(auxiliary, auxiliary)
    if closed and not history:
        lines.append(
            "\nOther retained stages: "
            + ", ".join(f"{n} {state}" for state, n in sorted(closed.items()))
            + ". Use --history for details."
        )
    storage = ledger.get("storage_plan", {})
    plan_path = storage.get("plan")
    if plan_path and (ROOT / plan_path).exists():
        plan = json.loads((ROOT / plan_path).read_text())
        generation_applied = storage.get("runtime_reserve_applied", False)
        reserve_key = (
            "minimum_free_bytes_proposed" if generation_applied else "minimum_free_bytes_current"
        )
        cap_state = "approved" if plan.get("user_approved") else "proposed"
        lines.extend(
            [
                "",
                f"{'Retained storage budget (paused)' if paused else 'Local storage'}: "
                f"{storage.get('status', plan['status'])} | "
                f"{shutil.disk_usage(ROOT).free / 1e9:.1f} GB free | "
                f"active reserve {plan[reserve_key] / 1e9:.0f} GB | "
                f"additional artifact cap {cap_state}"
                f" {plan['requested_additional_artifact_cap_bytes'] / 1e9:.0f} GB",
            ]
        )
        guardian = storage.get("guardian")
        if guardian and (ROOT / guardian / "progress.json").exists():
            state = json.loads((ROOT / guardian / "progress.json").read_text())
            guardian_age = time.time() - state["updated_at"]
            guardian_status = state["status"]
            if paused:
                guardian_status = "historical snapshot; project paused"
            elif guardian_status == "running" and guardian_age > 20:
                guardian_status = "STALE (resume after host-process check)"
            lines.append(
                f"Storage guardian {guardian_status}: "
                f"{state['net_consumed_bytes'] / 1e9:.2f}/{state['cap_bytes'] / 1e9:.0f} GB "
                f"net new space | approved future minimum "
                f"{state['minimum_free_bytes'] / 1e9:.0f} GB | "
                f"heartbeat {guardian_age:.0f}s ago"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--history", action="store_true", help="Show all retained closed stages")
    args = parser.parse_args()
    try:
        while True:
            print(
                snapshot(args.history) if args.once else "\033[2J\033[H" + snapshot(args.history),
                flush=True,
            )
            if args.once:
                return
            time.sleep(2)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()

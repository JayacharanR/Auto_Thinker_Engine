#!/usr/bin/env python3
"""Candidate-once frozen Town05 evaluation; registration and qualification required."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_encounters import settle_group  # noqa: E402
from src.navigation.final_audit import (  # noqa: E402
    conservative_final_counts,
    final_opportunity_audit,
    final_setup_audit,
    raw_attempt_events,
    recorder_attempt_contacts,
)
from src.navigation.final_evaluation import (  # noqa: E402
    SUITE,
    candidate_request,
    digest,
    register_candidate,
)
from src.navigation.metrics import wilson_interval  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--expert", action="store_true")
    args = parser.parse_args()
    if args.expert == bool(args.checkpoint):
        parser.error("Choose expert reference or a qualified camera checkpoint")
    key = digest(args.checkpoint) if args.checkpoint else "expert_reference_v1"
    root = ROOT / "outputs/navigation_m6_final" / key
    registration = ROOT / "outputs/navigation_final_registry" / f"{key}.json"
    request = candidate_request(args.checkpoint, args.qualification, root, args.expert)
    root.mkdir(parents=True, exist_ok=True)
    with RunLock(ROOT / "outputs/.navigation.lock"), RunLock(root / ".runner.lock"):
        register_candidate(registration, request)
        atomic_json(root / "request.json", request)
        # A completed candidate is reported, never executed a second time.
        if (root / "summary.json").exists():
            prior = json.loads((root / "summary.json").read_text())
            if prior["status"] == "complete":
                print(json.dumps(prior, indent=2))
                return
        with BudgetWatch(root):
            records, attempts = {}, {}
            atomic_json(root / "progress.json", dict(status="running", stage="M6 frozen final",
                       frames=0, target_frames=36, updated_at=time.time()))
            for seed in request["seeds"]:
                options = SimpleNamespace(
                    out=root,
                    seed=seed,
                    chunk_wall=21600,
                    checkpoint=args.checkpoint,
                    rule_overrides=False,
                    checkpoint_sha256=key,
                    final_registration=registration,
                )
                base = root / f"seed_{seed}" / "all_routes"
                base.parent.mkdir(exist_ok=True)
                a, r = settle_group(options, SUITE / "Town05_Opt.xml", request["route_ids"], base)
                attempts[str(seed)] = a
                records.update({f"{seed}:{rid}": entry for rid, entry in r.items()})
                atomic_json(
                    root / "progress.json",
                    dict(
                        status="running",
                        stage="M6 frozen final",
                        frames=len(records),
                        target_frames=36,
                        current_seed=seed,
                        updated_at=time.time(),
                    ),
                )
                if (root / "STOP").exists():
                    break
        suite = json.loads((SUITE / "suite.json").read_text())
        inventory_path = ROOT / "outputs/navigation_final_geometry_20261018/Town05_Opt.json"
        if digest(inventory_path) != suite["inventory_sha256"]:
            raise ValueError("Registered final physical inventory changed")
        setups = final_setup_audit(
            root, suite, records, attempts, json.loads(inventory_path.read_text())
        )
        native_events = raw_attempt_events(root)
        contacts = recorder_attempt_contacts(root, records, attempts, request["seeds"])
        opportunities = final_opportunity_audit(
            root, suite, records, attempts, json.loads(inventory_path.read_text())
        )
        # A planned stop never physically encountered is unverified, not a
        # compliant pass inferred from a zero native violation counter.
        setups = {
            key: dict(
                setup,
                valid=setup["valid"]
                and opportunities["final_missions"][key]["required_stop_observed"],
                required_stop_observed=opportunities["final_missions"][key][
                    "required_stop_observed"
                ],
            )
            for key, setup in setups.items()
        }
        counts = conservative_final_counts(records, setups, 36, contacts["final_contacts"])
        safe, rule_safe = counts["safe_completed"], counts["rule_safe_completed"]
        missing = [
            f"{s}:{r}"
            for s in request["seeds"]
            for r in request["route_ids"]
            if f"{s}:{r}" not in records
        ]
        status = (
            "stopped" if (root / "STOP").exists() else "complete" if not missing else "incomplete"
        )
        summary = dict(
            status=status,
            expected_missions=36,
            recorded_missions=len(records),
            safe_completed=safe,
            rule_safe_completed=rule_safe,
            all_collisions=contacts["all_attempt_ego_contact_starts"],
            native_all_attempt_collision_events=native_events["collisions"],
            red_violations=native_events["red_violations"],
            stop_violations=native_events["stop_violations"],
            collision_contact_audit=contacts,
            physical_rule_opportunity_audit=opportunities,
            declared_rule_coverage_observed=opportunities["declared_coverage_observed"],
            native_attempt_event_audit=native_events,
            final_setup_audit=setups,
            conservative_counts=counts,
            native_raw_safe_completed=sum(e["safe_completed"] for e in records.values()),
            native_raw_rule_safe_completed=sum(e["rule_safe_completed"] for e in records.values()),
            safe_completion_fraction=safe / 36,
            safe_completion_wilson95=wilson_interval(safe, 36),
            missing_missions=missing,
            records=records,
            all_attempts=attempts,
            note=(
                "Missing/invalid setups or incomplete recorder evidence remain failed "
                "in36 denominator. "
                "All recorder contacts across all attempts counted without fault/speed filtering. "
                "Cluster by route for inferential comparisons; Wilson assumes independence."
            ),
        )
        atomic_json(root / "summary.json", summary)
        atomic_json(
            root / "progress.json",
            dict(
                status=status,
                stage="M6 frozen final",
                frames=len(records),
                target_frames=36,
                updated_at=time.time(),
            ),
        )
        print(
            json.dumps(
                {k: v for k, v in summary.items() if k not in ("records", "all_attempts")}, indent=2
            )
        )


if __name__ == "__main__":
    main()

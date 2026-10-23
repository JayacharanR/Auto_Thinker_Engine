"""Immutable final-candidate registration and frozen Town05 request validation."""

import hashlib
import json
from pathlib import Path

from src.navigation.recording_policy import frozen_showcase_policy

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "configs/routes/town05_final_v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_final_suite(root=SUITE):
    root = Path(root).resolve()
    if root != SUITE.resolve():
        raise ValueError("Only the registered Town05_final_v2 suite is authorized")
    expected = (root / "suite.sha256").read_text().split()[0]
    suite = json.loads((root / "suite.json").read_text())
    if (
        digest(root / "suite.json") != expected
        or digest(root / "Town05_Opt.xml") != suite["route_sha256"]
    ):
        raise ValueError("Frozen final suite/XML changed")
    if not suite["frozen"] or suite["role"] != "final_test" or len(suite["routes"]) != 12:
        raise ValueError("Incomplete final registration")
    if suite["traffic_seeds"] != [3031, 4031, 5031] or not suite["count_all_collisions"]:
        raise ValueError("Final seed/scoring contract changed")
    return suite, expected


def candidate_request(checkpoint, evidence, evaluation_root, expert=False):
    suite, sha = verify_final_suite()
    request = dict(
        schema="navigation-final-candidate-v1",
        suite=str(SUITE.resolve()),
        suite_sha256=sha,
        evaluation_root=str(Path(evaluation_root).resolve()),
        policy_kind="expert" if expert else "student",
        seeds=suite["traffic_seeds"],
        route_ids=sorted(r["route_id"] for r in suite["routes"]),
        count_all_collisions=True,
        rule_overrides=False,
        candidate_once=True,
        recording_policy=frozen_showcase_policy(),
        inference_source_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                *sorted((ROOT / "src/navigation").glob("*.py")),
                ROOT / "scripts/run_navigation.py",
                ROOT / "scripts/run_navigation_final.py",
                ROOT / "scripts/run_encounters.py",
                ROOT / "patches/navigation_garage.patch",
                ROOT / "configs/navigation_vendor.json",
                ROOT / "configs/navigation.yaml",
            ]
        },
    )
    if expert:
        if checkpoint or evidence:
            raise ValueError("Expert reference has no student checkpoint or training evidence")
    else:
        if not checkpoint or not evidence:
            raise ValueError(
                "Final student registration needs a checkpoint and M2–M5 qualification evidence"
            )
        path = Path(evidence).resolve()
        qualified = json.loads(path.read_text())
        checkpoint = Path(checkpoint).resolve()
        if qualified.get("candidate_sha256") != digest(checkpoint):
            raise ValueError("Qualification evidence belongs to a different candidate")
        if qualified.get("town05_used_for_training_or_selection") is not False:
            raise ValueError("Explicit Town05 training/selection exclusion required")
        for stage in ("M2", "M3", "M4", "M5"):
            item = qualified.get("milestones", {}).get(stage, {})
            if (
                not item.get("evidence_complete")
                or not item.get("gate_passed")
                or not item.get("reports")
            ):
                raise ValueError(f"Final driving remains gated by {stage}")
            for report in item["reports"]:
                p = Path(report["path"])
                if digest(p) != report["sha256"]:
                    raise ValueError(f"{stage} evidence file changed")
        import torch

        from src.navigation.policy_input_contract import (
            LEGACY_SCHEMA,
            policy_input_schema,
        )

        schema = policy_input_schema(torch.load(checkpoint, map_location="cpu", weights_only=True))
        if qualified.get("policy_input_schema", LEGACY_SCHEMA) != schema:
            raise ValueError("Qualification input schema differs from the candidate")
        request.update(
            policy_input_schema=schema,
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint),
            qualification=str(path),
            qualification_sha256=digest(path),
        )
    return request


def register_candidate(path, request):
    """The same candidate has one immutable registry and one resumable result root."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != request:
            raise ValueError("Existing candidate registration cannot be changed")
        return request
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create is the candidate-once claim. Serialize without a replace.
    with path.open("x") as stream:
        json.dump(request, stream, indent=2)
        stream.write("\n")
    return request


def validate_final_request(
    registration,
    route,
    output,
    seed,
    subset,
    student,
    collection,
    seed_per_route=False,
    max_steps=0,
):
    suite, sha = verify_final_suite()
    request = json.loads(Path(registration).read_text())
    if request.get("recording_policy") != frozen_showcase_policy():
        raise ValueError("Registered final recording-only policy changed")
    if any(digest(ROOT / p) != h for p, h in request["inference_source_sha256"].items()):
        raise ValueError("Registered inference/evaluation source changed")
    if request["suite_sha256"] != sha or request["suite"] != str(SUITE.resolve()):
        raise ValueError("Final candidate registered against another suite")
    if Path(route).resolve() != (SUITE / "Town05_Opt.xml").resolve():
        raise ValueError("Final run must use the exact registered route XML")
    try:
        Path(output).resolve().relative_to(Path(request["evaluation_root"]).resolve())
    except ValueError:
        raise ValueError("Candidate has one immutable final result directory") from None
    if seed not in suite["traffic_seeds"] or seed_per_route or max_steps or collection:
        raise ValueError(
            "Final runs require exact traffic seeds, complete routes and no collection"
        )
    if not subset or any(int(r) not in request["route_ids"] for r in subset.split(",")):
        raise ValueError("Explicit registered final route IDs required")
    if request["policy_kind"] == "student":
        if (
            not student
            or student["sha256"] != request["checkpoint_sha256"]
            or student["rule_overrides"]
            or student.get("policy_input_schema", "navigation-sensor-packet-v1")
            != request["policy_input_schema"]
        ):
            raise ValueError("Final candidate/override identity differs from registration")
        if digest(request["qualification"]) != request["qualification_sha256"]:
            raise ValueError("Final qualification evidence changed")
        if (
            candidate_request(
                request["checkpoint"], request["qualification"], request["evaluation_root"]
            )
            != request
        ):
            raise ValueError("Final qualification sources changed")
    elif student or request["policy_kind"] != "expert":
        raise ValueError("Final expert/student identity changed")
    return dict(
        registration=str(Path(registration).resolve()),
        registration_sha256=digest(registration),
        suite_sha256=sha,
        candidate_once=True,
        recording_policy=request["recording_policy"],
    )

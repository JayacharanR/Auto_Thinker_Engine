#!/usr/bin/env python3
"""Score every planned override-disabled rule trial, including missing attempts."""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_rule_prototypes import assess  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa: E402
from src.navigation.rule_behavior import behavior  # noqa: E402
from src.navigation.rule_suite import FAMILIES, SEEDS, SETUP_RUNTIME_SOURCES  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines()]


def certificate_matches_execution(case, output):
    """Bind to archived execution sources, allowing later read-only assessment."""
    path = Path(case["case"])
    certificate_path = path.parent / "setup_certificate.json"
    if (
        digest(path) != case["case_sha256"]
        or digest(certificate_path) != case["certificate_sha256"]
    ):
        return False
    certificate = read(certificate_path)
    if (
        certificate.get("schema") != "navigation-physical-setup-certificate-v2"
        or certificate.get("case_sha256") != digest(path)
        or len(certificate.get("paired_case_sha256", {})) != 2
        or certificate["paired_case_sha256"].get(str(path.resolve())) != digest(path)
        or not certificate.get("evidence_sha256")
        or set(certificate.get("runtime_source_sha256", {})) != set(SETUP_RUNTIME_SOURCES)
        or certificate.get("expert_success_used_to_select_case") is not False
    ):
        return False
    for key in (
        "synchronized_frames", "declared_actor_setup", "physical_signal_binding",
        "one_factor_pair_verified", "prefix_bounds_verified", "source_hashes_verified",
    ):
        if certificate.get(key) is not True:
            return False
    for hashes in (certificate["paired_case_sha256"], certificate["evidence_sha256"]):
        if any(digest(p) != value for p, value in hashes.items()):
            return False
    archived = read(output / "source_hashes.json")
    manifest = read(output / "manifest.json")
    for source, expected in certificate["runtime_source_sha256"].items():
        if source == "configs/navigation.yaml":
            if manifest["config_sha256"] != expected:
                return False
        elif archived.get(source) != expected or digest(output / "source" / source) != expected:
            return False
    return True


def summarize(root, actuation_path):
    root = Path(root).resolve()
    request = read(root / "request.json")
    trials = request["trials"]
    if (
        len(trials) != 72
        or set(r["seed"] for r in trials) != set(SEEDS)
        or set(r["family"] for r in trials) != set(FAMILIES)
        or len({r["name"] for r in trials}) != 72
        or request["rule_overrides"] is not False
    ):
        raise ValueError("The full registered72-trial denominator must remain intact")
    identities = {(r["family"], r["location"], r["arm"], r["seed"]) for r in trials}
    if len(identities) != 72:
        raise ValueError("Duplicate physical case/seed identities cannot fill the denominator")
    for family in FAMILIES:
        locations = {r["location"] for r in trials if r["family"] == family}
        if len(locations) != 2 or any(
            (family, location, arm, seed) not in identities
            for location in locations for arm in (0, 1) for seed in SEEDS
        ):
            raise ValueError("Each family requires both arms at two locations and three seeds")
    actions = read(actuation_path)
    action_rows = {str(Path(row["output"]).resolve()): row for row in actions["records"]}
    outcomes = read(root / "outcomes.json") if (root / "outcomes.json").exists() else []
    if len({r["name"] for r in outcomes}) != len(outcomes):
        raise ValueError("A candidate/case/seed cannot silently replace a prior outcome")
    rows, all_contacts, recorder_errors = [], [], []
    considered = set()
    for trial in trials:
        output = root / trial["name"]
        partial = []
        for recording in sorted((output / "recordings").glob("*.log")):
            considered.add(recording.resolve())
            try:
                contact = recorder_contacts(recording)
                partial.append(contact)
                all_contacts.append(dict(trial=trial["name"], **contact))
            except Exception as error:
                recorder_errors.append(
                    dict(trial=trial["name"], path=str(recording), error=str(error))
                )
        row = dict(**trial, output=str(output), passed=False, evidence_complete=False)
        try:
            if not (output / "summary.json").exists():
                raise ValueError("Missing terminal native evidence; retained failed in72")
            native = read(output / "summary.json")
            manifest = read(output / "manifest.json")
            student = manifest.get("student")
            if (
                not student
                or student["rule_overrides"] is not False
                or student["sha256"] != request["checkpoint_sha256"]
                or digest(output / "policy.pt") != request["checkpoint_sha256"]
                or manifest["seed"] != trial["seed"]
                or student["architecture"] != request["architecture"]
                or native.get("harness_crashed")
                or len(native["episodes"]) != 1
            ):
                raise ValueError("Not the declared terminal, override-disabled camera execution")
            frozen_spec = read(trial["case"])
            if manifest["rule_test"]["case_sha256"] != trial["case_sha256"]:
                raise ValueError("Actual controlled intervention differs from frozen case")
            audited = assess(output)
            actual_sources = read(output / "source_hashes.json")
            if any(
                actual_sources.get(path) != expected
                for path, expected in request["inference_source_sha256"].items()
            ):
                raise ValueError("Native execution differs from the source-frozen candidate")
            if not all(
                audited[key]
                for key in (
                    "source_hashes_verified", "synchronized_frames", "declared_actor_setup",
                    "evaluated_controls_unchanged", "prefix_never_reactivated",
                )
            ):
                raise ValueError("Physical execution/control/video audit did not pass")
            if not certificate_matches_execution(trial, output):
                raise ValueError("Physical certificate does not bind actual archived execution")
            action = action_rows.get(str(output))
            if (
                not action
                or action["differences"]
                or action["missing_final_ack_frames"]
                or not action["checked_acknowledgments"]
                or action["recorder_sha256"] != audited["recorder"]["sha256"]
                or not audited["recorder"]["complete"]
            ):
                raise ValueError("Independent complete native next-tick actuation proof missing")
            episode = next((output / "episodes").iterdir())
            control_log = episode / "controlled_rule/requested_executed.jsonl"
            if digest(control_log) != action["control_log_sha256"]:
                raise ValueError("Independent actuation/control log identity changed")
            scored = behavior(
                frozen_spec,
                lines(episode / "controlled_rule/setup_trace.jsonl"),
                native["episodes"][0],
                audited["recorder"],
                setup_validated=True,
            )
            row.update(
                passed=scored["passed"],
                evidence_complete=True,
                behavior=scored,
                physical_audit=audited,
                actuation_audit=action,
            )
        except Exception as error:
            row["evidence_error"] = str(error)
            row["partial_known_contact_starts"] = sum(p["ego_contact_starts"] for p in partial)
        rows.append(row)
    unexpected = []
    for recording in sorted(root.rglob("*.log")):
        if recording.parent.name != "recordings" or recording.resolve() in considered:
            continue
        unexpected.append(str(recording))
        try:
            all_contacts.append(dict(trial="unregistered_attempt", **recorder_contacts(recording)))
        except Exception as error:
            recorder_errors.append(dict(path=str(recording), error=str(error)))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    complete = sum(r["evidence_complete"] for r in rows)
    passed = sum(r["passed"] for r in rows)
    return dict(
        status="complete" if complete == 72 else "incomplete_evidence",
        planned_trials=72,
        evidence_complete_trials=complete,
        passed_trials=passed,
        failed_or_unverified_trials=72 - passed,
        gate_passed=complete == passed == 72 and not recorder_errors and not unexpected,
        unfiltered_all_attempt_contact_starts=sum(r["ego_contact_starts"] for r in all_contacts),
        contacts_lower_bound=complete != 72 or bool(recorder_errors)
        or any(not r["complete"] for r in all_contacts),
        recorder_errors=recorder_errors,
        unexpected_attempt_recorders=unexpected,
        all_attempt_recorders=all_contacts,
        family_counts={
            family: dict(planned=12, passed=sum(r["passed"] for r in values))
            for family, values in grouped.items()
        },
        rows=rows,
        checkpoint_sha256=request["checkpoint_sha256"],
        request_sha256=digest(root / "request.json"),
        actuation_audit_sha256=digest(actuation_path),
        rule_overrides=False,
        autonomous_route_success_claim=False,
        note="Component interventions only. Expert outcomes never count as student passes. "
        "Missing/invalid evidence stays failed in72; every native contact remains counted.",
    )


def grid(result):
    text = [
        "# Override-disabled camera rule tests", "",
        f"{result['passed_trials']}/72 pass; "
        f"{result['evidence_complete_trials']}/72 complete evidence. "
        f"All-attempt contacts: {result['unfiltered_all_attempt_contact_starts']}"
        + (" (lower bound)." if result["contacts_lower_bound"] else "."), "",
        "Each cell lists seeds61001/61002/61003. P=pass, F=behavior failure, "
        "?=missing or invalid evidence. Scripted prefixes are not autonomous driving.", "",
        "| Family | First location arm0 | First location arm1 | Second location arm0 | "
        "Second location arm1 | Passes |",
        "|---|---|---|---|---|---|",
    ]
    for family in FAMILIES:
        values = [r for r in result["rows"] if r["family"] == family]
        locations = sorted({r["location"] for r in values})
        cells = []
        for location in locations:
            for arm in (0, 1):
                ordered = sorted(
                    (r for r in values if r["location"] == location and r["arm"] == arm),
                    key=lambda r: r["seed"],
                )
                cells.append(" / ".join(
                    "P" if r["passed"] else "F" if r["evidence_complete"] else "?"
                    for r in ordered
                ))
        text.append(f"| {family} | " + " | ".join(cells) +
                    f" | {result['family_counts'][family]['passed']}/12 |")
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--actuation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Retain every prior student rule assessment")
    args.out.mkdir(parents=True)
    for path in (Path(__file__), ROOT / "src/navigation/rule_behavior.py"):
        shutil.copyfile(path, args.out / path.name)
    result = summarize(args.root, args.actuation)
    result["scoring_source_sha256"] = {
        p.name: digest(p) for p in args.out.glob("*.py")
    }
    atomic_json(args.out / "assessment.json", result)
    (args.out / "grid.md").write_text(grid(result))
    print(json.dumps({k: result[k] for k in (
        "status", "passed_trials", "planned_trials", "unfiltered_all_attempt_contact_starts",
        "gate_passed",
    )}), flush=True)


if __name__ == "__main__":
    main()

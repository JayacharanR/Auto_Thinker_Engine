#!/usr/bin/env python3
"""Recompute full planned-denominator encounter reference with recorder contacts."""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_encounters import planned_headline, settled, setup_audit  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--contact-audit", type=Path, required=True)
    parser.add_argument("--rpc-verification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.out.exists():
        raise FileExistsError("Keep prior contact reference analysis")
    request = json.loads((args.run / "request.json").read_text())
    catalog_path = Path(request["catalog"]) / "catalog.json"
    if digest(catalog_path) != request["catalog_sha256"]:
        raise ValueError("Reference catalog changed")
    catalog = json.loads(catalog_path.read_text())
    finals, attempts = settled(args.run, catalog)
    setup = setup_audit(args.run, catalog, finals, attempts)
    audit = json.loads(args.contact_audit.read_text())
    rpc = json.loads(args.rpc_verification.read_text())
    if not rpc["all_complete_recordings_match"]:
        raise ValueError("Official query cross-check failed")
    checked = {row["path"]: row["sha256"] for row in rpc["checked"] if row["all_rows_match"]}
    recordings = {row["path"]: row for row in audit["records"]}
    expected = list(args.run.glob("*/chunk_*/recordings/route_*_rep*.log"))
    if {str(p.resolve()) for p in expected} != set(recordings):
        raise ValueError("Not every attempt recording audited")
    adjusted = copy.deepcopy(finals)
    detail = {}
    missing = []
    for rid, record in finals.items():
        town = next(e["town"] for e in catalog["encounters"] if e["route_id"] == rid)
        final_attempt = next(
            row["attempt"]
            for row in reversed(attempts[rid])
            if row.get("route_id") == record["route_id"] and "crash" not in row["status"].lower()
        )
        path = str(
            (args.run / town / final_attempt / "recordings" / f"route_{rid}_rep0.log").resolve()
        )
        contact = recordings.get(path)
        certified = bool(contact and contact["complete"] and checked.get(path) == contact["sha256"])
        if not certified:
            missing.append(rid)
        safe = certified and contact["ego_contact_starts"] == 0
        adjusted[rid]["safe_completed"] = record["safe_completed"] and safe
        adjusted[rid]["rule_safe_completed"] = record["rule_safe_completed"] and safe
        detail[rid] = dict(
            recorder=path,
            complete_and_rpc_verified=certified,
            recorder_contact_starts=contact["ego_contact_starts"] if contact else None,
            native_collision_events=record["collisions"],
            native_rule_safe=record["rule_safe_completed"],
            recorder_rule_safe=adjusted[rid]["rule_safe_completed"],
        )
    all_contacts = sum(r["ego_contact_starts"] for r in recordings.values())
    headline = planned_headline(catalog, adjusted, setup, all_contacts)
    families = {}
    for family in sorted({e["family"] for e in catalog["encounters"]}):
        subset = dict(
            catalog, encounters=[e for e in catalog["encounters"] if e["family"] == family]
        )
        ids = {e["route_id"] for e in subset["encounters"]}
        family_contacts = sum(
            row["ego_contact_starts"]
            for path, row in recordings.items()
            if int(Path(path).stem.split("_")[1]) in ids
        )
        families[family] = planned_headline(
            subset,
            {k: v for k, v in adjusted.items() if k in ids},
            {k: v for k, v in setup.items() if k in ids},
            family_contacts,
        )
    output = dict(
        schema="recorder-contact-reference-v1",
        run=str(args.run),
        headline=headline,
        families=families,
        final_contact_evidence=detail,
        incomplete_final_contact_evidence=missing,
        all_attempt_partial_recordings=[
            r["path"] for r in recordings.values() if not r["complete"]
        ],
        native_results_preserved=True,
        all_attempt_contacts=all_contacts,
        contact_audit_sha256=digest(args.contact_audit),
        official_query_verification_sha256=digest(args.rpc_verification),
        catalog_sha256=digest(catalog_path),
        script_sha256=digest(__file__),
        note=(
            'Every planned mission/invalid setup retained; all recorded '
            'contacts, no fault/speed filter. Partial recordings give '
            'lower-bound contacts only.'
        ),
    )
    atomic_json(args.out, output)
    print(json.dumps(headline), flush=True)


if __name__ == "__main__":
    main()

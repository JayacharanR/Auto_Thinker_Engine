#!/usr/bin/env python3
"""Freeze a user-approved train/held-out junction split without changing dev_v1."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_encounter_catalog import annotate, route_geometry, write_town_xml  # noqa: E402
from src.navigation.catalog import coverage, movement_key, select_catalog, turn  # noqa: E402
from src.navigation.encounters import xyz  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.splits import partition_locations  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def freeze_catalog(out, picks, maps, planners, carla, split, role, inventory, hashes, vendor):
    picks.sort(key=lambda p: (p["movement"]["town"], p["family"]))
    entries = []
    for rid, p in enumerate(picks):
        m, town = p["movement"], p["movement"]["town"]
        g = route_geometry(carla, maps[town], planners[town], p)
        trace = planners[town].trace_route(g["start"].location, g["end"].location)
        crossed = sorted({w.junction_id for w, _ in trace if w.is_junction})
        if not set(crossed) <= set(split[town][role]):
            raise ValueError("Generated route crosses the other location partition")
        entries.append(
            {
                "route_id": rid,
                "town": town,
                "family": p["family"],
                "scenario": p["scenario"],
                "parameters": p["parameters"],
                "occurrence": p["occurrence"],
                "movement_key": [str(x) for x in movement_key(m)],
                "junction_id": m["junction_id"],
                "crossed_junction_ids": crossed,
                "turn": turn(m),
                "control": m["control"],
                "signal_ids_at_inventory": m["signal_ids"],
                "stop_ids_at_inventory": m["stop_ids"],
                "start_xyz": xyz(g["start"].location),
                "end_xyz": xyz(g["end"].location),
                "trigger_xyz": xyz(g["trigger"].location),
                "trigger_yaw": g["trigger"].rotation.yaw,
                "planned_length_m": g["length_m"],
            }
        )
    out.mkdir(parents=True, exist_ok=False)
    towns = {}
    for town in sorted(maps):
        group = [e for e in entries if e["town"] == town]
        if not group:
            continue
        path = out / f"{town}.xml"
        write_town_xml(path, town, group)
        ids = [e["route_id"] for e in group]
        towns[town] = {"route_ids": [min(ids), max(ids)], "count": len(ids)}
        atomic_json(
            path.with_suffix(".json"),
            {
                "purpose": f"M1 spatial partition: {role}",
                "town": town,
                "route_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "lengths_m": {str(e["route_id"]): e["planned_length_m"] for e in group},
                "length_m": None,
            },
        )
    record = {
        "version": "location_split_v1",
        "frozen": True,
        "role": role,
        "selection_seed": 20261015,
        "selection_rule": "geometry-only seeded location partition",
        "location_reuse_rule": "repeated locations; cluster intervals by junction",
        "inventory": str(inventory),
        "inventory_sha256": hashes,
        "vendor": vendor,
        "towns": towns,
        "coverage": coverage(picks),
        "encounters": entries,
    }
    atomic_json(out / "catalog.json", record)
    digest = hashlib.sha256((out / "catalog.json").read_bytes()).hexdigest()
    (out / "catalog.sha256").write_text(digest + "  catalog.json\n")
    return {"sha256": digest, "coverage": record["coverage"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="configs/encounters/location_split_v1")
    parser.add_argument("--inventory", default="outputs/navigation_encounter_inventory_v2_20261009")
    parser.add_argument("--carla-root", default="~/software/CARLA_0.9.15")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Never overwrite a frozen split")
    vendor = activate(args.carla_root)
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    config = yaml.safe_load((ROOT / "configs/navigation.yaml").read_text())
    maps, planners, movements, hashes = {}, {}, [], {}
    for town in config["data"]["train_towns"]:
        name = town + "_Opt"
        path = Path(args.inventory) / f"{name}.json"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        record = json.loads(path.read_text())
        xodr = (
            Path(args.carla_root).expanduser()
            / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
        )
        world_map = carla.Map(name, xodr.read_text())
        if (
            hashlib.sha256(world_map.to_opendrive().encode()).hexdigest()
            != record["opendrive_sha256"]
        ):
            raise ValueError("Map differs from reference inventory")
        planner = GlobalRoutePlanner(world_map, 1.0)
        annotate(carla, world_map, planner, record, name)
        for m in record["movements"]:
            if m["valid"] and not m.get("excluded"):
                trace = planner.trace_route(
                    carla.Location(*m["start_xyz"]), carla.Location(*m["end_xyz"])
                )
                m["crossed_junction_ids"] = sorted(
                    {w.junction_id for w, _ in trace if w.is_junction}
                )
        maps[name], planners[name] = world_map, planner
        movements.extend(record["movements"])
    split, train, dev, attempt = partition_locations(movements)
    catalogs = {}
    for role, pool in (("train", train), ("heldout_dev", dev)):
        picks = select_catalog(pool, config["evaluation"]["families"], 50, 20261015)
        catalogs[role] = freeze_catalog(
            out / role,
            picks,
            maps,
            planners,
            carla,
            split,
            role,
            Path(args.inventory),
            hashes,
            vendor,
        )
    atomic_json(
        out / "locations.json",
        {
            "seed": 20261015,
            "selection_attempt": attempt,
            "fraction": 0.2,
            "rule": "first seeded partition with geometry coverage; no driving outcomes",
            "assignments": split,
            "catalogs": catalogs,
            "approval": "User approved new location split before student training, 2026-10-15",
            "legacy_dev_v1": "Preserved expert calibration; "
            "report shared-location status explicitly",
        },
    )
    (out / "locations.sha256").write_text(
        hashlib.sha256((out / "locations.json").read_bytes()).hexdigest() + "\n"
    )
    print(json.dumps({"assignments": split, "selection_attempt": attempt}, indent=2))


if __name__ == "__main__":
    main()

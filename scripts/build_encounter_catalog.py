#!/usr/bin/env python3
"""Build and freeze the development encounter catalog from a runtime inventory.

Offline only: OpenDRIVE maps and the harness's own route planner, no simulator.
Selection is seeded and outcome-blind (src/navigation/catalog.py). Movements whose
planned route crosses a stop trigger that covers only junction-internal lanes are
excluded by rule, because the evaluator would demand a stop inside the junction.
Refuses to overwrite an existing catalog; Town05 is never read.
"""

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.catalog import (  # noqa: E402
    HARD_BREAK_TRIGGER_M,
    RELEASE_START_M,
    coverage,
    movement_key,
    select_catalog,
    turn,
)
from src.navigation.encounters import lane_key, path_length, xyz  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402

SELECTION_SEED = 20261009
WEATHER = {"route_percentage": "0", "cloudiness": "0", "precipitation": "0",
           "sun_altitude_angle": "70"}  # fmt: skip


def location(carla, point):
    return carla.Location(*point)


def first_junction_waypoint(wp, limit=5.0, step=0.5):
    # Stored coordinates are rounded to 1 mm, which can place the re-found entry
    # just upstream of the original, so a single 1 m step may stop short.
    travelled = 0.0
    while not wp.is_junction and travelled < limit:
        nxt = wp.next(step)
        if not nxt:
            return None
        wp, travelled = nxt[0], travelled + step
    return wp if wp.is_junction else None


def annotate(carla, carla_map, planner, record, town):
    from srunner.tools.scenario_helper import filter_junction_wp_direction, get_junction_topology

    internal_stop_lanes = {
        tuple(lane["lane"])
        for stop in record["stops"]
        if not any(lane["leads_into_junction"] for lane in stop["affected_lanes"])
        for lane in stop["affected_lanes"]
    }
    for m in record["movements"]:
        m["town"] = town
        first = first_junction_waypoint(carla_map.get_waypoint(location(carla, m["entry_xyz"])))
        if first is None:
            m["valid"], m["invalid_reason"] = False, "offline entry does not reach junction"
            continue
        entries, _ = get_junction_topology(first.get_junction())
        m["has_opposite_entry"] = bool(filter_junction_wp_direction(first, entries, "opposite"))
        m["has_left_entry"] = bool(filter_junction_wp_direction(first, entries, "left"))
        traced = planner.trace_route(location(carla, m["start_xyz"]), location(carla, m["end_xyz"]))
        crossed = {tuple(lane_key(w)) for w, _ in traced} & internal_stop_lanes
        m["excluded"] = bool(crossed)
        m["excluded_reason"] = "crosses junction-internal-only stop trigger" if crossed else None


def route_geometry(carla, carla_map, planner, pick):
    m = pick["movement"]
    start = carla_map.get_waypoint(location(carla, m["start_xyz"]))
    if pick["family"] == "signal_release_memory":
        entry = carla_map.get_waypoint(location(carla, m["entry_xyz"]))
        start = entry.previous(RELEASE_START_M)[0]
    end = carla_map.get_waypoint(location(carla, m["end_xyz"]))
    traced = planner.trace_route(start.transform.location, end.transform.location)
    points = [xyz(w.transform.location) for w, _ in traced]
    if not any(w.is_junction and w.get_junction().id == m["junction_id"] for w, _ in traced):
        raise RuntimeError(f"Planned route misses junction {m['junction_id']} in {m['town']}")
    trigger = start
    if pick["scenario"] == "HardBreakRoute":
        trigger = start.next(HARD_BREAK_TRIGGER_M)[0]
        if trigger.is_junction:
            raise RuntimeError("HardBreakRoute trigger fell inside the junction")
    return {
        "start": start.transform,
        "end": end.transform,
        "trigger": trigger.transform,
        "length_m": round(path_length(points), 3),
    }


def write_town_xml(path, town, entries):
    routes = ET.Element("routes")
    for e in entries:
        node = ET.SubElement(routes, "route", id=str(e["route_id"]), town=town)
        ET.SubElement(ET.SubElement(node, "weathers"), "weather", **WEATHER)
        waypoints = ET.SubElement(node, "waypoints")
        for name in ("start_xyz", "end_xyz"):
            x, y, z = e[name]
            ET.SubElement(waypoints, "position", x=f"{x:.5f}", y=f"{y:.5f}", z=f"{z:.5f}")
        scenarios = ET.SubElement(node, "scenarios")
        if e["scenario"]:
            scenario = ET.SubElement(
                scenarios, "scenario", name=f"{e['scenario']}_{e['route_id']}", type=e["scenario"]
            )
            x, y, z = e["trigger_xyz"]
            ET.SubElement(
                scenario, "trigger_point",
                x=f"{x:.5f}", y=f"{y:.5f}", z=f"{z:.5f}", yaw=f"{e['trigger_yaw']:.3f}",
            )  # fmt: skip
            for key, value in e["parameters"].items():
                if isinstance(value, list):
                    ET.SubElement(scenario, key, **{"from": str(value[0]), "to": str(value[1])})
                else:
                    ET.SubElement(scenario, key, value=str(value))
    ET.indent(routes)
    ET.ElementTree(routes).write(path, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/navigation.yaml")
    parser.add_argument(
        "--carla-root", default=os.environ.get("CARLA_ROOT", "~/software/CARLA_0.9.15")
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", default="configs/encounters/dev_v1")
    args = parser.parse_args()
    vendor = activate(args.carla_root)
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    config = yaml.safe_load(Path(args.config).read_text())
    evaluation = config["evaluation"]
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to replace frozen catalog {out}")
    inventory = Path(args.inventory)
    maps, planners, movements, inventory_hashes = {}, {}, [], {}
    for town in config["data"]["train_towns"]:
        name = f"{town}_Opt"
        path = inventory / f"{name}.json"
        inventory_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        record = json.loads(path.read_text())
        xodr = (
            Path(args.carla_root).expanduser()
            / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
        )
        carla_map = carla.Map(town, xodr.read_text())
        if (
            hashlib.sha256(carla_map.to_opendrive().encode()).hexdigest()
            != record["opendrive_sha256"]
        ):
            raise RuntimeError(f"Offline {town} OpenDRIVE differs from the runtime inventory map")
        planner = GlobalRoutePlanner(carla_map, 1.0)
        annotate(carla, carla_map, planner, record, name)
        maps[name], planners[name] = carla_map, planner
        movements.extend(record["movements"])

    catalog = select_catalog(
        movements, evaluation["families"], evaluation["encounters_per_family"], SELECTION_SEED
    )
    catalog.sort(key=lambda p: (p["movement"]["town"], evaluation["families"].index(p["family"])))
    entries = []
    for route_id, pick in enumerate(catalog):
        m, town = pick["movement"], pick["movement"]["town"]
        geometry = route_geometry(carla, maps[town], planners[town], pick)
        entries.append(
            {
                "route_id": route_id,
                "town": town,
                "family": pick["family"],
                "scenario": pick["scenario"],
                "parameters": pick["parameters"],
                "occurrence": pick["occurrence"],
                "movement_key": [str(x) for x in movement_key(m)],
                "junction_id": m["junction_id"],
                "turn": turn(m),
                "control": m["control"],
                "signal_ids_at_inventory": m["signal_ids"],
                "stop_ids_at_inventory": m["stop_ids"],
                "start_xyz": xyz(geometry["start"].location),
                "end_xyz": xyz(geometry["end"].location),
                "trigger_xyz": xyz(geometry["trigger"].location),
                "trigger_yaw": geometry["trigger"].rotation.yaw,
                "planned_length_m": geometry["length_m"],
            }
        )
    out.mkdir(parents=True)
    towns = {}
    for town in dict.fromkeys(e["town"] for e in entries):
        town_entries = [e for e in entries if e["town"] == town]
        xml = out / f"{town}.xml"
        write_town_xml(xml, town, town_entries)
        ids = [e["route_id"] for e in town_entries]
        towns[town] = {"route_ids": [min(ids), max(ids)], "count": len(ids)}
        atomic_json(
            xml.with_suffix(".json"),
            {
                "purpose": "M0/M1 development encounter catalog; never Town05",
                "town": town,
                "route_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
                "lengths_m": {str(e["route_id"]): e["planned_length_m"] for e in town_entries},
                "length_m": None,
            },
        )
    excluded = sum(bool(m.get("excluded")) for m in movements)
    catalog_record = {
        "version": "dev_v1",
        "frozen": True,
        "selection_seed": SELECTION_SEED,
        "selection_rule": "seeded round-robin over strata; prefer unused movements/lights/"
        "junctions; reuse only when a stratum is exhausted; never uses driving outcomes",
        "exclusion_rule": "movements whose planned route crosses a stop trigger covering only "
        "junction-internal lanes",
        "excluded_movements": excluded,
        "traffic_seed_rule": "run with --seed BASE --seed-per-route: seed = BASE + route_id",
        "location_reuse_rule": "stop-sign family reuses locations across seeds; intervals "
        "must be clustered by junction/stop location",
        "inventory": str(inventory),
        "inventory_sha256": inventory_hashes,
        "vendor": vendor,
        "towns": towns,
        "coverage": coverage(catalog),
        "encounters": entries,
    }
    atomic_json(out / "catalog.json", catalog_record)
    digest = hashlib.sha256((out / "catalog.json").read_bytes()).hexdigest()
    (out / "catalog.sha256").write_text(digest + "  catalog.json\n")
    print(json.dumps({"catalog_sha256": digest, "towns": towns, "excluded": excluded,
                      "coverage": catalog_record["coverage"]}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()

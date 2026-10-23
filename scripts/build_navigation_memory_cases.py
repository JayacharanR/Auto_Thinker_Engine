#!/usr/bin/env python3
"""Outcome-blind memory geometries at the first two eligible TRAIN junctions."""

import argparse
import ast
import copy
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.rule_suite import arms  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Preserve every memory-setup prototype")
    catalog = json.loads(
        (ROOT / "configs/encounters/location_split_v1/train/catalog.json").read_text()
    )
    search = ROOT / "reports/navigation_memory_geometry_search_20261019.json"
    eligible = json.loads(search.read_text())["eligible"]
    entries = {e["route_id"]: e for e in catalog["encounters"]}
    chosen, seen = [], set()
    for row in sorted(eligible, key=lambda r: r["route_id"]):
        entry = entries[row["route_id"]]
        key = entry["town"], entry["junction_id"]
        if key not in seen:
            chosen.append(entry)
            seen.add(key)
        if len(chosen) == 2:
            break
    if [e["route_id"] for e in chosen] != [79, 80]:
        raise ValueError("Memory repair differs from prospectively declared geometry")
    vendor = activate()
    import carla

    args.out.mkdir(parents=True)
    sources = [
        Path(__file__),
        search,
        ROOT / "reports/navigation_memory_setup_repair_protocol_20261019.md",
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    cases = []
    for index, entry in enumerate(chosen):
        prototype = (
            ROOT
            / "outputs/navigation_rule_cases_v3_20261019/signal1_out_of_view_memory_a0/case.json"
        )
        common = json.loads(prototype.read_text())
        town = entry["town"]
        inventory = json.loads((ROOT / catalog["inventory"] / f"{town}.json").read_text())
        signal = next(
            s for s in inventory["lights"] if s["id"] == entry["signal_ids_at_inventory"][0]
        )
        lane = tuple(ast.literal_eval(entry["movement_key"][2]))
        line = next(row for row in signal["stop_lanes"] if tuple(row["lane"]) == lane)
        world_map = carla.Map(
            town,
            (
                Path(vendor["carla_root"]) / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
            ).read_text(),
        )
        direction = world_map.get_waypoint(
            carla.Location(*line["xyz"])
        ).transform.get_forward_vector()
        common.update(
            source_catalog_route_id=entry["route_id"],
            source_entry=entry,
            crossed_junction_ids=entry["crossed_junction_ids"],
            signal_reference=signal,
            line_xyz=line["xyz"],
            line_heading_xy=[direction.x, direction.y],
        )
        for treatment in arms("out_of_view_memory"):
            folder = args.out / f"memory{index}_out_of_view_memory_a{treatment['arm']}"
            folder.mkdir()
            tree = ET.Element("routes")
            node = ET.SubElement(tree, "route", id="0", town=town)
            ET.SubElement(
                ET.SubElement(node, "weathers"),
                "weather",
                route_percentage="0",
                cloudiness="0",
                precipitation="0",
                sun_altitude_angle="70",
            )
            waypoints = ET.SubElement(node, "waypoints")
            for endpoint in (entry["start_xyz"], entry["end_xyz"]):
                ET.SubElement(waypoints, "position", **dict(zip("xyz", map(str, endpoint))))
            ET.SubElement(node, "scenarios")
            ET.indent(tree)
            route_path = folder / "route.xml"
            ET.ElementTree(tree).write(route_path, encoding="utf-8", xml_declaration=True)
            # Route length is independently checked by the normal harness; its
            # catalog value has the same2%/10m tolerance as original prototypes.
            atomic_json(
                folder / "route.json",
                dict(
                    purpose="Separate controlled memory setup, not canonical benchmark",
                    town=town,
                    route_sha256=digest(route_path),
                    length_m=entry["planned_length_m"],
                    junction_ids=entry["crossed_junction_ids"],
                ),
            )
            spec = dict(copy.deepcopy(common), treatment=treatment, route_sha256=digest(route_path))
            atomic_json(folder / "case.json", spec)
            (folder / "case.sha256").write_text(digest(folder / "case.json") + "\n")
            cases.append(
                dict(
                    path=str(folder / "case.json"),
                    sha256=digest(folder / "case.json"),
                    family="out_of_view_memory",
                    location=f"memory{index}",
                    arm=treatment["arm"],
                )
            )
    atomic_json(
        args.out / "cases.json",
        dict(
            cases=cases,
            geometry_search_sha256=digest(search),
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            actual_towns=["Town02_Opt"],
            independent_junctions=[242, 132],
            setup_certified=False,
            student_tests_started=False,
        ),
    )
    print(json.dumps(dict(case_geometries=len(cases), setup_certified=False)), flush=True)


if __name__ == "__main__":
    main()

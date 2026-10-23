"""Geometry-only location partitions, frozen before student collection/training."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.navigation.catalog import family_pool


def nearest_junction(route_xyz, junction_xyz, junction_ids):
    """Spatial clearance also catches coincident road endpoints with different IDs."""
    route, junctions = np.asarray(route_xyz), np.asarray(junction_xyz)
    if not len(route) or not len(junctions) or len(junctions) != len(junction_ids):
        raise ValueError("Need route and identified junction geometry")
    if not np.isfinite(route).all() or not np.isfinite(junctions).all():
        raise ValueError("Nonfinite partition geometry")
    best, index = float("inf"), None
    for offset in range(0, len(route), 32):
        distances = np.sum((route[offset : offset + 32, None] - junctions[None]) ** 2, axis=2)
        flat = int(distances.argmin())
        value = float(distances.flat[flat])
        if value < best:
            best, index = value, flat % len(junctions)
    return {"clearance_m": math.sqrt(best), "junction_id": int(junction_ids[index])}


def collection_geometry_eligibility(catalog_dir, carla_root=None, additional_reserved=None):
    """Conservative training-only filter; never edit or resample frozen evaluation routes."""
    from src.navigation.collection import verify_collection_catalog
    from src.navigation.vendor import activate

    directory = Path(catalog_dir).resolve()
    catalog = json.loads((directory / "catalog.json").read_text())
    for town in catalog["towns"]:
        verify_collection_catalog(directory, directory / f"{town}.xml")
    split = json.loads((directory.parent / "locations.json").read_text())
    vendor = activate(carla_root)
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    kept, excluded, maps = [], [], {}
    for town in sorted(catalog["towns"]):
        inventory = Path(catalog["inventory"]) / f"{town}.json"
        if not inventory.is_absolute():
            inventory = Path(__file__).resolve().parents[2] / inventory
        payload = inventory.read_bytes()
        if hashlib.sha256(payload).hexdigest() != catalog["inventory_sha256"][town]:
            raise ValueError("Collection geometry inventory changed")
        record = json.loads(payload)
        xodr = (
            Path(vendor["carla_root"])
            / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{town.removesuffix('_Opt')}.xodr"
        )
        world_map = carla.Map(town, xodr.read_text())
        digest = hashlib.sha256(world_map.to_opendrive().encode()).hexdigest()
        if digest != record["opendrive_sha256"]:
            raise ValueError("Collection geometry map changed")
        maps[town] = digest
        heldout = set(split["assignments"][town]["heldout_dev"])
        heldout.update((additional_reserved or {}).get(town, []))
        junctions = [
            w
            for w in world_map.generate_waypoints(0.5)
            if w.is_junction and w.junction_id in heldout
        ]
        coordinates = [
            [w.transform.location.x, w.transform.location.y, w.transform.location.z]
            for w in junctions
        ]
        ids = [w.junction_id for w in junctions]
        planner = GlobalRoutePlanner(world_map, 1.0)
        for entry in catalog["encounters"]:
            if entry["town"] != town:
                continue
            traced = planner.trace_route(
                carla.Location(*entry["start_xyz"]), carla.Location(*entry["end_xyz"])
            )
            points = [
                [w.transform.location.x, w.transform.location.y, w.transform.location.z]
                for w, _ in traced
            ]
            closest = nearest_junction(points, coordinates, ids)
            if closest["clearance_m"] < 1.0:
                excluded.append(
                    {
                        "route_id": entry["route_id"],
                        "town": town,
                        "family": entry["family"],
                        **closest,
                    }
                )
            else:
                kept.append(entry["route_id"])

    def strata(rows):
        return {(e["town"], e["family"]) for e in rows}

    retained = [e for e in catalog["encounters"] if e["route_id"] in kept]
    if additional_reserved is None and strata(retained) != strata(catalog["encounters"]):
        raise ValueError("Spatial clearance removed a collection coverage stratum")
    return {
        "rule": "Training only: at least 1m centreline clearance from held-out junction geometry",
        "route_sampling_m": 1.0,
        "junction_sampling_m": 0.5,
        "minimum_clearance_m": 1.0,
        "map_sha256": maps,
        "kept_route_ids": sorted(kept),
        "excluded": excluded,
        **({"additional_reserved_junctions": additional_reserved} if additional_reserved else {}),
        "scope": "Geometry only; no driving outcomes. Runtime partition guard remains active. "
        "No frozen catalog, held-out route, location assignment or evaluation metric changed.",
    }


def partition_locations(movements, seed=20261015, heldout_fraction=0.2):
    usable = [m for m in movements if m["valid"] and not m.get("excluded")]
    towns = sorted({m["town"] for m in usable})
    locations = {t: sorted({m["junction_id"] for m in usable if m["town"] == t}) for t in towns}
    rng = np.random.default_rng(seed)
    families = [
        "signal_stop",
        "signal_release_memory",
        "stop_sign",
        "crossing_yield",
        "following_stopped_vehicle",
        "route_junction",
    ]

    def pool(heldout, chosen):
        return [
            m
            for m in usable
            if (m["junction_id"] in heldout[m["town"]]) == chosen
            and all(
                (j in heldout[m["town"]]) == chosen
                for j in m.get("crossed_junction_ids", [m["junction_id"]])
            )
        ]

    for attempt in range(1000):
        heldout = {
            t: set(
                int(i)
                for i in rng.choice(
                    ids, size=max(1, math.ceil(len(ids) * heldout_fraction)), replace=False
                )
            )
            for t, ids in locations.items()
        }
        train, dev = pool(heldout, False), pool(heldout, True)
        if any(not family_pool(f, p) for f in families for p in (train, dev)):
            continue
        # Both partitions must support signals and following across all five towns.
        if any(
            {m["town"] for m in family_pool(f, p)} != set(towns)
            for f in ("signal_stop", "following_stopped_vehicle")
            for p in (train, dev)
        ):
            continue
        # At least two distinct stop junctions per partition; no outcome information.
        if any(
            len({(m["town"], m["junction_id"]) for m in family_pool("stop_sign", p)}) < 2
            for p in (train, dev)
        ):
            continue
        assignments = {
            t: {"train": [j for j in ids if j not in heldout[t]], "heldout_dev": sorted(heldout[t])}
            for t, ids in locations.items()
        }
        return assignments, train, dev, attempt
    raise RuntimeError("No geometry-valid partition with the declared coverage constraints")

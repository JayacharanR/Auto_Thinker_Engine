"""Training-only rule geometry. Frustum overlap is not visible-light ground truth."""

import math
from types import SimpleNamespace

import numpy as np


def transform_matrix(pose):
    """CARLA/Unreal x-forward, y-right, z-up; pose xyz,pitch,yaw,roll (degrees)."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("A finite full six-dimensional pose is required")
    pitch, yaw, roll = np.deg2rad(pose[3:])
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    result = np.eye(4)
    result[:3, :3] = [
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ]
    result[:3, 3] = pose[:3]
    return result


def camera_matrix(ego_pose, calibration):
    return transform_matrix(ego_pose) @ transform_matrix(
        [*calibration.xyz, *calibration.pitch_yaw_roll]
    )


def project_points(points, ego_pose, calibration, near=0.05):
    """Return pixels and forward depth; behind/near points have NaN pixels."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("World points must be finite Nx3")
    if near <= 0:
        raise ValueError("Positive near plane required")
    local = (
        np.linalg.inv(camera_matrix(ego_pose, calibration)) @ np.c_[points, np.ones(len(points))].T
    ).T
    # UE x,y,z -> conventional optical y,-z,x.
    optical = local[:, [1, 2, 0]] * [1, -1, 1]
    depth = optical[:, 2]
    pixels = np.full((len(points), 2), np.nan)
    valid = depth > near
    projected = optical[valid] @ calibration.intrinsic.T
    pixels[valid] = projected[:, :2] / projected[:, 2:]
    return pixels, depth


def project_box(vertices, ego_pose, calibration):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (8, 3):
        raise ValueError("Eight world-space bounding-box vertices required")
    pixels, depth = project_points(vertices, ego_pose, calibration)
    valid = np.isfinite(pixels).all(axis=1)
    if not valid.all():
        return {
            "status": "behind_near_plane" if not valid.any() else "crosses_near_plane",
            "in_image": False if not valid.any() else None,
            "bbox_xyxy": None,
            "height_px": None,
            "width_px": None,
            "minimum_depth_m": float(depth.min()),
        }
    x0, y0 = pixels.min(axis=0)
    x1, y1 = pixels.max(axis=0)
    overlap = bool(x1 > 0 and x0 < calibration.width and y1 > 0 and y0 < calibration.height)
    return {
        "status": "projected",
        "in_image": overlap,
        "fully_in_image": bool(
            x0 >= 0 and y0 >= 0 and x1 <= calibration.width and y1 <= calibration.height
        ),
        "bbox_xyxy": [float(v) for v in (x0, y0, x1, y1)],
        "height_px": float(y1 - y0),
        "width_px": float(x1 - x0),
        "minimum_depth_m": float(depth.min()),
    }


def match_light(actor, geometry):
    """Actor IDs are episode-local; join servers by full static pose, never state."""
    matches = []
    for row in geometry["lights"]:
        first, second = np.asarray(actor["pose"]), np.asarray(row["pose"])
        if (
            np.linalg.norm(first[:3] - second[:3]) < 0.001
            and np.max(np.abs((first[3:] - second[3:] + 180) % 360 - 180)) < 0.01
        ):
            matches.append(row)
    if actor["type"] != "traffic.traffic_light" or len(matches) != 1:
        raise ValueError("Applicable lamp has no unique full-pose geometry match")
    return matches[0]


def ego_pose_from_facts(facts):
    """Recorded world snapshot retains pitch/roll omitted by the planar teacher pose."""
    heroes = [
        a
        for a in facts["world"]["actors"]
        if a["role"] == "hero" and a["type"].startswith("vehicle.")
    ]
    if len(heroes) != 1:
        raise ValueError("Exactly one recorded ego required")
    pose = heroes[0]["pose"]
    if not np.allclose(pose[:3], facts["ego_pose_world"][:3], rtol=0, atol=0.001):
        raise ValueError("Same-frame ego/world pose mismatch")
    if abs((pose[4] - math.degrees(facts["ego_pose_world"][3]) + 180) % 360 - 180) > 0.01:
        raise ValueError("Same-frame ego/world heading mismatch")
    return pose


def project_rule_sample(facts, calibrations, geometry):
    if not isinstance(facts["world"]["frame"], int):
        raise ValueError("Recorded synchronous world frame required")
    pose = ego_pose_from_facts(facts)
    light_id = facts["applicable_light_id"]
    if light_id is None:
        return None
    actors = [a for a in facts["world"]["actors"] if a["id"] == light_id]
    if len(actors) != 1 or actors[0].get("signal") != facts["next_light_state"]:
        raise ValueError("Applicable light identity/state mismatch")
    row = match_light(actors[0], geometry)
    views = {}
    for name, c in calibrations.items():
        boxes = [project_box(b["vertices"], pose, c) for b in row["boxes_world"]]
        stops = []
        for line in row["evaluator_stop_lines"]:
            pixels, _ = project_points(line["world_segment"], pose, c)
            stops.append(
                {
                    "lane": line["lane"],
                    "pixels": pixels.tolist() if np.isfinite(pixels).all() else None,
                }
            )
        in_image = any(b["in_image"] is True for b in boxes)
        if not boxes or (not in_image and any(b["in_image"] is None for b in boxes)):
            in_image = None
        views[name] = {
            "boxes": boxes,
            "housing_in_frustum": in_image,
            "evaluator_line_candidates": stops,
        }
    return {
        "recorded_actor_id": light_id,
        "geometry_capture_actor_id": row["capture_actor_id"],
        "opendrive_id": row["opendrive_id"],
        "views": views,
        "visibility_valid": None,
        "state_supervision_approved": False,
        "stop_line_lane_selection": "unassigned; all actor lanes, no painted-line claim",
    }


def capture_light_geometry(world):
    """Privileged static metadata only; never call from learned inference."""
    import carla
    from srunner.scenariomanager.scenarioatomics.atomic_criteria import RunningRedLightTest

    helper = SimpleNamespace(
        _map=world.get_map(),
        rotate_point=lambda point, angle: RunningRedLightTest.rotate_point(None, point, angle),
    )

    def xyz(point):
        return [point.x, point.y, point.z]

    def stop_line(wp, half_width):
        side = wp.transform.get_right_vector() * (half_width * wp.lane_width)
        center = wp.transform.location
        return {
            "lane": [wp.road_id, wp.section_id, wp.lane_id],
            "center": xyz(center),
            "world_segment": [xyz(center - side), xyz(center + side)],
            "half_width_in_lane_widths": half_width,
        }

    result = []
    for light in sorted(world.get_actors().filter("traffic.traffic_light*"), key=lambda a: a.id):
        pose = light.get_transform()
        _, evaluator_waypoints = RunningRedLightTest.get_traffic_light_waypoints(helper, light)
        boxes = light.get_light_boxes()
        # CARLA 0.9.15 GetTrafficLightBoundingBox applies component-to-world already.
        # Applying the actor pose again would incorrectly transform these twice.
        result.append(
            {
                "capture_actor_id": light.id,
                "opendrive_id": light.get_opendrive_id(),
                "pose": [
                    *xyz(pose.location),
                    pose.rotation.pitch,
                    pose.rotation.yaw,
                    pose.rotation.roll,
                ],
                "boxes_world": [
                    {
                        "center": xyz(box.location),
                        "extent": xyz(box.extent),
                        "rotation": [box.rotation.pitch, box.rotation.yaw, box.rotation.roll],
                        "vertices": [xyz(p) for p in box.get_world_vertices(carla.Transform())],
                    }
                    for box in boxes
                ],
                "api_stop_positions": [stop_line(wp, 0.5) for wp in light.get_stop_waypoints()],
                "evaluator_stop_lines": [stop_line(wp, 0.6) for wp in evaluator_waypoints],
            }
        )
    return result


def stopped_runs(rows, threshold=0.1):
    """Contiguous 10Hz teacher-stop events with their immediate release, if recorded."""
    result, start = [], None
    for i in range(len(rows) + 1):
        stop = i < len(rows) and rows[i]["teacher_target_speed_mps"] < threshold
        if stop and start is None:
            start = i
        if not stop and start is not None:
            duration = rows[i - 1]["timestamp"] - rows[start]["timestamp"] + 0.1
            result.append(
                {
                    "start_frame": rows[start]["frame"],
                    "end_frame": rows[i - 1]["frame"],
                    "samples": i - start,
                    "duration_s": duration,
                    "longer_than_3_2s": duration > 3.20001,
                    "release_observed": i < len(rows),
                    "release_frame": rows[i]["frame"] if i < len(rows) else None,
                    "cause_at_start": rows[start]["teacher_hazards"],
                }
            )
            start = None
    return result

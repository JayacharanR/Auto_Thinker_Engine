"""
CARLA Smoke Test — Gate 0 verification.

Must pass before any training begins. Verifies:
(a) CARLA server launches and connects
(b) Vehicle spawns successfully
(c) Simulator steps 100 times with random actions
(d) Reports steps/sec and peak VRAM

Target: complete in under 2 minutes.
Baseline throughput number logged here is the reference for
detecting regressions in later phases.
"""

import argparse
import os
import sys
import time

import numpy as np


def run_smoke_test(host: str = "localhost", port: int = 2000) -> dict:
    """
    Execute the CARLA smoke test.

    Returns:
        Dict with throughput, VRAM, and pass/fail status.
    """
    import carla
    import torch

    results = {
        "passed": False,
        "steps_per_sec": 0.0,
        "peak_vram_mb": 0.0,
        "total_steps": 0,
        "errors": [],
    }

    client = None
    vehicle = None

    try:
        # (a) Connect to CARLA server
        print(f"Connecting to CARLA at {host}:{port}...")
        client = carla.Client(host, port)
        client.set_timeout(30.0)

        world = client.get_world()
        print(f"  Connected to: {world.get_map().name}")

        # Configure for headless, low quality
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        world.apply_settings(settings)

        # (b) Spawn one vehicle
        print("Spawning vehicle...")
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            results["errors"].append("No spawn points available")
            return results

        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"  Vehicle spawned: {vehicle.type_id} (id={vehicle.id})")

        # Attach camera sensor (to simulate Phase 1 observation setup)
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "128")
        camera_bp.set_attribute("image_size_y", "128")
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-15),
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        frame_received = [False]
        def on_image(image):
            frame_received[0] = True

        camera.listen(on_image)

        # Reset VRAM tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # (c) Step simulator 100 times with random actions
        print("Stepping simulator (100 steps with random actions)...")
        num_steps = 100
        start_time = time.time()

        for step in range(num_steps):
            # Random control
            control = carla.VehicleControl()
            control.steer = np.random.uniform(-0.3, 0.3)
            control.throttle = np.random.uniform(0.3, 0.7)
            control.brake = 0.0
            vehicle.apply_control(control)

            world.tick()

        elapsed = time.time() - start_time
        results["total_steps"] = num_steps

        # (d) Report metrics
        steps_per_sec = num_steps / elapsed
        results["steps_per_sec"] = steps_per_sec

        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1e6
            results["peak_vram_mb"] = peak_vram
        else:
            results["peak_vram_mb"] = 0.0

        results["camera_working"] = frame_received[0]
        results["passed"] = True

        # Print summary
        print("\n" + "=" * 60)
        print("SMOKE TEST RESULTS")
        print("=" * 60)
        print(f"  Status:          {'PASSED ✓' if results['passed'] else 'FAILED ✗'}")
        print(f"  Steps/sec:       {steps_per_sec:.1f}")
        print(f"  Peak VRAM:       {results['peak_vram_mb']:.1f} MB")
        print(f"  Total steps:     {num_steps}")
        print(f"  Elapsed time:    {elapsed:.1f}s")
        print(f"  Camera working:  {results['camera_working']}")
        print(f"  Target (< 2min): {'YES' if elapsed < 120 else 'NO'}")
        print("=" * 60)

        # Cleanup
        camera.stop()
        camera.destroy()

    except Exception as e:
        results["errors"].append(str(e))
        print(f"\nSMOKE TEST FAILED: {e}")

    finally:
        if vehicle is not None:
            try:
                vehicle.destroy()
            except Exception:
                pass

        # Reset world settings
        if client is not None:
            try:
                world = client.get_world()
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
            except Exception:
                pass

    return results


def main():
    parser = argparse.ArgumentParser(description="CARLA Smoke Test (Gate 0)")
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    args = parser.parse_args()

    results = run_smoke_test(host=args.host, port=args.port)

    if not results["passed"]:
        print("\n⚠️  Smoke test FAILED. Do NOT proceed to Phase 1.")
        if results["errors"]:
            print("  Errors:")
            for err in results["errors"]:
                print(f"    - {err}")
        sys.exit(1)
    else:
        print("\n✓ Smoke test PASSED. Ready for Phase 1.")
        sys.exit(0)


if __name__ == "__main__":
    main()

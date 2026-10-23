"""Bounded cleanup of an owned native collector; preserve its original failure."""

import time

from src.navigation.runtime import atomic_json


def cleanup_owned_collection(server, client, world, rig, actors, output, recording=False):
    """Never issue RPCs to an exited owned server or raise over a primary error.

    The surrounding CarlaServer context still owns normal server shutdown. This
    helper sends no process signals, starts no process and cannot stop another
    queue. Writers and local logs must be closed separately by the caller.
    """
    errors = []
    process = server.process
    if process is None:
        raise ValueError("Cleanup requires the exact owned CarlaServer process")
    exited = process.poll()
    receipt = dict(
        server_returncode=exited,
        rpc_skipped_server_exited=exited is not None,
        sensor_handles=len(rig.sensors) if rig else 0,
        actor_handles=len(actors),
        rpc_timeout_seconds=20.0,
        errors=errors,
        primary_failure_preserved=True,
        updated_at=time.time(),
    )
    if exited is None:

        def attempt(name, action):
            try:
                action()
            except Exception as error:
                errors.append(dict(operation=name, error=str(error)))

        attempt("normal_cleanup_RPC_timeout", lambda: client.set_timeout(20.0))
        if recording:
            attempt("stop_owned_recorder", client.stop_recorder)

        # Resume physics before normal resource teardown, as the established
        # collector does, without waiting on any external owner.
        def advance_world():
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        sensors = list(rig.sensors) if rig else []
        seen = set()
        for actor in sensors + list(reversed(actors)):
            if actor.id in seen:
                continue
            seen.add(actor.id)
            if actor.type_id.startswith("sensor."):
                attempt(f"stop_sensor_{actor.id}", actor.stop)
            attempt(f"destroy_actor_{actor.id}", actor.destroy)
        # Match the established collector: stop sensor callbacks and destroy
        # actors BEFORE resuming asynchronous physics and normal engine exit.
        attempt("resume_owned_world", advance_world)
        # CarlaServer establishes20sfor normal work. A short teardown bound is
        # scoped to cleanup and must not leak into the next map reload.
        attempt("restore_normal_client_timeout", lambda: client.set_timeout(20.0))
        receipt["normal_client_timeout_restored"] = not any(
            e["operation"] == "restore_normal_client_timeout" for e in errors
        )
    if rig:
        rig.sensors.clear()
    atomic_json(output / "native_cleanup.json", receipt)
    return receipt

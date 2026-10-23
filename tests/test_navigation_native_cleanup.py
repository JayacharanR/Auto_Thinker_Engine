"""A dead native server must not cause N sequential20second cleanup RPC timeouts."""

from types import SimpleNamespace

from src.navigation.native_cleanup import cleanup_owned_collection


class Unreachable:
    def __getattr__(self, name):
        raise AssertionError(f"No RPC/actor access permitted after server exit: {name}")


def test_exited_server_skips_all_rpc_and_still_preserves_receipt(tmp_path):
    rig = SimpleNamespace(sensors=[Unreachable()] * 9)
    server = SimpleNamespace(process=SimpleNamespace(poll=lambda: -11))
    result = cleanup_owned_collection(
        server,
        Unreachable(),
        Unreachable(),
        rig,
        [Unreachable(), Unreachable()],
        tmp_path,
        recording=True,
    )
    assert result["rpc_skipped_server_exited"] and result["server_returncode"] == -11
    assert result["errors"] == [] and rig.sensors == []
    assert (tmp_path / "native_cleanup.json").exists()


def test_live_cleanup_continues_after_an_actor_failure_without_hiding_it(tmp_path):
    destroyed, timeouts = [], []

    def fail():
        raise RuntimeError("owned sensor stop failed")

    sensor = SimpleNamespace(
        id=1, type_id="sensor.camera.rgb", stop=fail, destroy=lambda: destroyed.append(1)
    )
    actor = SimpleNamespace(id=2, type_id="vehicle.car", destroy=lambda: destroyed.append(2))
    server = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
    client = SimpleNamespace(set_timeout=timeouts.append, stop_recorder=lambda: None)
    settings = SimpleNamespace(synchronous_mode=True, fixed_delta_seconds=0.05)
    world = SimpleNamespace(get_settings=lambda: settings, apply_settings=lambda s: None)
    result = cleanup_owned_collection(
        server,
        client,
        world,
        SimpleNamespace(sensors=[sensor]),
        [actor, sensor],
        tmp_path,
        recording=True,
    )
    assert timeouts == [20.0, 20.0] and destroyed == [1, 2]
    assert result["normal_client_timeout_restored"]
    assert not settings.synchronous_mode and settings.fixed_delta_seconds is None
    assert len(result["errors"]) == 1 and result["primary_failure_preserved"]

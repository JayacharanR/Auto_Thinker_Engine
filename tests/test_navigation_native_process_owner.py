"""A supervisor must refuse foreign/reused processes and require its STOP file."""

import os

import pytest

from src.navigation.native_process_owner import NativeEngineOwner, require_process_identity


def test_process_identity_checks_uid_group_binary_port_and_reuse():
    saved = dict(pid=123, binary="/owned/Carla", port=2100, create_time=1.5)
    command = ["/owned/Carla", "-carla-rpc-port=2100"]
    require_process_identity(saved, command, os.getuid(), 123, 1.5)
    for changed in (
        (command, os.getuid() + 1, 123, 1.5),
        (["/other/Carla", command[1]], os.getuid(), 123, 1.5),
        ([command[0], "-carla-rpc-port=2200"], os.getuid(), 123, 1.5),
        (command, os.getuid(), 99, 1.5),
        (command, os.getuid(), 123, 100.0),
    ):
        with pytest.raises(ValueError, match="identity changed"):
            require_process_identity(saved, *changed)


def test_stop_request_is_required_before_any_process_or_signal_access(tmp_path):
    owner = NativeEngineOwner(tmp_path, tmp_path / "case", "Town01_Opt")
    with pytest.raises(ValueError, match="STOPpermits|STOP permits"):
        owner.stop_after_request()
    (tmp_path / "case").mkdir()
    (tmp_path / "case/STOP").touch()
    with pytest.raises(ValueError, match="No direct owned"):
        owner.stop_after_request()

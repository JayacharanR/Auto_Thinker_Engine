"""Candidate identity and the actual encounter subprocess command boundary."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_encounters import run_chunk


def test_student_command_preserves_native_route_seed_and_off_overrides(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"immutable test candidate")
    args = SimpleNamespace(
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        seed=42,
        out=tmp_path,
        chunk_wall=1800,
        rule_overrides=False,
    )
    commands = []

    class Child:
        returncode = 0

        def poll(self):
            return 0

    def popen(command, **kwargs):
        commands.append(command)
        return Child()

    monkeypatch.setattr("scripts.run_encounters.subprocess.Popen", popen)
    assert run_chunk(args, Path("frozen.xml"), "10-19", tmp_path / "attempt", slot=1) == 0
    c = commands[0]
    assert c[3] == "student" and "--rule-overrides" not in c
    assert c[c.index("--checkpoint") + 1] == str(checkpoint)
    assert "--seed-per-route" in c and c[c.index("--seed") + 1] == "42"
    checkpoint.write_bytes(b"different")
    with pytest.raises(ValueError, match="changed"):
        run_chunk(args, Path("frozen.xml"), "10-19", tmp_path / "other")
    assert len(commands) == 1


def test_default_stays_expert_and_uses_no_student_control_flag(tmp_path, monkeypatch):
    args = SimpleNamespace(seed=42, out=tmp_path, chunk_wall=1800)
    commands = []
    monkeypatch.setattr(
        "scripts.run_encounters.subprocess.Popen",
        lambda c, **kw: commands.append(c) or SimpleNamespace(poll=lambda: 0, returncode=0),
    )
    run_chunk(args, Path("frozen.xml"), "0-9", tmp_path / "expert")
    assert commands[0][3] == "expert" and "--checkpoint" not in commands[0]

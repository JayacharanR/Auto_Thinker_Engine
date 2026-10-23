"""A queued approved patch cannot overwrite newly changed or unexpected files."""

import hashlib

import pytest

from scripts.apply_navigation_feedback_generation import verify_generation


def test_feedback_activation_requires_exact_existing_bytes_and_absent_new_target(tmp_path):
    existing = tmp_path / "existing.py"
    existing.write_text("original\n")
    registered = {
        "targets": {
            "existing.py": {"before_sha256": hashlib.sha256(existing.read_bytes()).hexdigest()},
            "new.py": {"before_sha256": None},
        }
    }
    verify_generation(registered, tmp_path)
    existing.write_text("later user edit\n")
    with pytest.raises(ValueError, match="changed"):
        verify_generation(registered, tmp_path)
    assert existing.read_text() == "later user edit\n"
    existing.write_text("original\n")
    (tmp_path / "new.py").write_text("other work\n")
    with pytest.raises(ValueError, match="already exists"):
        verify_generation(registered, tmp_path)
    assert (tmp_path / "new.py").read_text() == "other work\n"

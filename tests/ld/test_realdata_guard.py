"""The realdata tier's check that the installed ogld names the checkout (ogld#5's review): an
editable install keeps the version it was built with, so a store built after the checkout moved
recorded another commit as its software.ogld."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent / "realdata" / "realdata_inputs.py"
_SPEC = importlib.util.spec_from_file_location("realdata_inputs_guard", _PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guard)

HEAD = "95e5f3e7f1a2b3c4d5e6f708192a3b4c5d6e7f80"


@pytest.mark.parametrize(
    ("version", "tags", "names"),
    [
        ("0.0.1.dev71+g95e5f3e7f", [], True),
        ("0.0.1.dev71+g95e5f3e7f.d20261110", [], True),
        ("0.0.1.dev57+g95a6c8276", [], False),
        ("0.2.0", ["v0.2.0"], True),
        ("0.2.0", ["v0.1.0"], False),
        ("0.0.0+unknown", [], False),
    ],
)
def test_an_installed_version_names_the_head_or_does_not(
    version: str, tags: list[str], names: bool
) -> None:
    assert guard.names_head(version, HEAD, tags) is names


def test_the_check_names_the_installed_version_and_the_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata
    import subprocess

    root = Path(__file__).resolve().parents[1]
    git = ["git", "-C", str(root), "rev-parse", "HEAD"]
    head = subprocess.run(git, capture_output=True, text=True, check=True).stdout.strip()
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.0.1.dev1+g000000000")
    message = guard.installed_mismatch()
    assert "0.0.1.dev1+g000000000" in message
    assert head[:12] in message
    assert "pixi reinstall --locked -e dev ogld" in message
    monkeypatch.setattr(importlib.metadata, "version", lambda _: f"0.0.1.dev1+g{head[:9]}")
    assert guard.installed_mismatch() == ""

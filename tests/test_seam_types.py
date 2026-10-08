"""The array seam type-checks against zarr-python 3's own types (#244 review, finding 4).

The package's mypy run keeps ``follow_imports = "skip"`` for zarr (see the
reason recorded in ``pyproject.toml``): following zarr's types adds 153 findings,
111 of them in ``opengwasdb/query/facade.py``, a typing refactor of the read paths
that #245 and #247 rewrite anyway. The cost is that mypy sees every zarr object
as ``Any``, so a misspelt or removed zarr API passes the baseline unnoticed.

`opengwasdb/store/arrays.py` is where the package's Store format meets zarr's API,
so it at least is checked with zarr's types followed: the project's own mypy
configuration with only the zarr skip removed, over the seam module alone. The
first run of this check found that the seam accepted mode ``"x"``, which zarr 3
rejects with a bare ``AssertionError``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

mypy_api = pytest.importorskip("mypy.api")

REPO = Path(__file__).resolve().parents[1]
SEAM = "opengwasdb/store/arrays.py"


def _config_following_zarr(tmp_path: Path) -> Path:
    """The project's mypy configuration, minus the override that skips zarr's types."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())
    mypy = project["tool"]["mypy"]
    overrides = [o for o in mypy.get("overrides", []) if o.get("follow_imports") != "skip"]
    assert len(overrides) == len(mypy.get("overrides", [])) - 1, "expected one skip override"
    lines = ["[mypy]"] + [
        f"{key} = {_ini(value)}" for key, value in mypy.items() if key != "overrides"
    ]
    for override in overrides:
        modules = override["module"]
        names = ",".join(modules) if isinstance(modules, list) else modules
        lines.append(f"[mypy-{names}]")
        lines += [f"{key} = {_ini(value)}" for key, value in override.items() if key != "module"]
    config = tmp_path / "mypy.ini"
    config.write_text("\n".join(lines) + "\n")
    return config


def _ini(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def test_the_seam_type_checks_against_zarr_types(tmp_path: Path) -> None:
    config = _config_following_zarr(tmp_path)
    assert "follow_imports" not in config.read_text(), (
        "zarr must be followed for this to mean anything"
    )
    stdout, stderr, status = mypy_api.run(
        ["--config-file", str(config), "--cache-dir", str(tmp_path / "cache"), str(REPO / SEAM)]
    )
    findings = [line for line in stdout.splitlines() if ": error:" in line]
    assert status == 0 and not findings, "\n".join(findings) or stderr or stdout

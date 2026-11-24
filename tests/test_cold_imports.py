"""Every entry point imports cleanly as the first thing a process does.

A circular import only fails when the cycle is entered from the wrong module, and
the suite almost always has imported `opengwasdb.model` by then. So each target
here gets a fresh interpreter. #464's move put `model` -> `readers` -> `model`
in the way of `readers.column_map`, `build.liftover` and the CLI's `resolve-analyses`.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "opengwasdb"

#: Modules a CLI command or a sibling repository imports first.
ENTRY_MODULES = (
    "opengwasdb.readers.column_map",
    "opengwasdb.build.liftover",
    "opengwasdb.store_reader",
    "opengwasdb.query",
    "opengwasdb.model",
    "opengwasdb.validation",
    "opengwasdb.cli.main",
)


def _subpackages() -> list[str]:
    return sorted(
        ".".join(init.parent.relative_to(PACKAGE.parent).parts)
        for init in PACKAGE.rglob("__init__.py")
    )


def _cold_import(module: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False
    )
    return done.stderr.strip().splitlines()[-1] if done.returncode else ""


def test_the_targets_are_meaningful() -> None:
    subpackages = _subpackages()
    assert len(subpackages) >= 15
    assert {"opengwasdb.model", "opengwasdb.readers", "opengwasdb.store_reader"} <= set(subpackages)
    for module in ENTRY_MODULES:
        assert (PACKAGE.parent / (module.replace(".", "/") + ".py")).exists() or (
            PACKAGE.parent / module.replace(".", "/") / "__init__.py"
        ).exists(), module


def test_every_entry_point_imports_cold() -> None:
    targets = sorted({"opengwasdb", *_subpackages(), *ENTRY_MODULES})
    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = {t: e for t, e in zip(targets, pool.map(_cold_import, targets)) if e}
    assert not failures, "\n".join(f"{t}: {e}" for t, e in failures.items())


@pytest.mark.parametrize("module", ["opengwasdb.readers"])
def test_resources_of_readers_resolve_cold(module: str) -> None:
    done = subprocess.run(
        [sys.executable, "-c", f"import importlib.resources as r; r.files({module!r})"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
